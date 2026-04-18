'''
preprocess_mri.py — DICOM to HDF5 preprocessing pipeline for cardiac MRI.

Main entry point for the cmr_toolkit preprocessing pipeline. Reads tar.gz archives
of DICOM studies and converts them into compressed HDF5 filestores organized by
patient (MRN) and scan (accession number). Each MRI series is stored as a 4D array
with associated metadata attributes.

Output structure:
    institution_MRN/
    └── AccessionNumber.h5
        ├── 4CH_FIESTA_BH       [f, c, h, w]   attrs: total_images, slice_frames
        ├── SAX_FIESTA_BH_1     [f, c, h, w]   attrs: total_images, slice_frames
        ├── SAX_FIESTA_BH_2     [f, c, h, w]   attrs: total_images, slice_frames
        └── STACK_LV3CH_FIESTA_BH [f, c, h, w] attrs: total_images, slice_frames

Supported institutions: stanford, ucsf, medstar, ukbiobank, upenn
Scales linearly to ~64 CPU cores. Can process 100k+ scans in under 3 hours.

Usage:
    python preprocess_mri.py -r /path/to/dicoms -o /path/to/output -i stanford -c 16
'''

import os
import numpy as np
import pydicom as dcm
import multiprocessing
import time
import pandas as pd
import csv
from shutil import move, rmtree
import glob
import tarfile
import argparse as ap
import matplotlib  
import h5py
import random
import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from natsort import natsorted, natsort_keygen
import bcolors
import pylibjpeg
from local_config import get_cfg, get_global_cfg
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from google.cloud import storage
from gcputils import wait_if_disk_full, GCP_Upload_Manager, mount_gcs_bucket, unmount_gcs_bucket

# Read and parse local_config.yaml and .env
load_dotenv()

_cfg        = get_cfg()
TMP_DIR     = _cfg.tmp_dir
BUCKET_NAME = get_global_cfg().bucket_name


### Global Functions ###

def notify_slack(message: str):
	'''
	Send a message to the configured Slack channel via cmr_bot.

	Reads SLACK_TOKEN and CHANNEL from environment variables (set via .env).
	Fails silently with a warning if the bot is not configured, so pipeline
	execution is never blocked by a missing Slack setup.

	Args:
		message: Text string to post to the Slack channel.
	'''
	try:
		slack_bot_token = os.getenv("SLACK_TOKEN")
		cmr_bot_channel = os.getenv("CHANNEL")
		app = App(token=slack_bot_token)
		app.client.chat_postMessage(channel=cmr_bot_channel, text=f"{message}")
		
	except Exception as ex:
		print(f'WARN: cmr_bot not configured, skipping...')
		print(ex)

class CMRI_PreProcessor:
	'''
	Cardiac MRI preprocessing pipeline: tar.gz DICOM archives → compressed HDF5.

	Orchestrates the full conversion workflow:
	    1. Extract tar.gz archive to TMP_DIR
	    2. Group DICOM files by SeriesDescription via view_disambugator()
	    3. Collate per-frame DICOMs into sorted 4D arrays via collate_arrays()
	    4. Write arrays to HDF5 with metadata attributes via array_to_h5()
	    5. Clean up extracted files from TMP_DIR

	Args:
		root_dir:            Path to directory containing .tgz DICOM archives, or GCS bucket path.
		output_dir:          Path where HDF5 output will be written.
		framesize:           Target frame size in pixels after resize (default: 480).
		institution_prefix:  Prefix string for output folders (e.g. 'stanford', 'ucsf').
		channels:            Storage mode — 'rgb' (3-channel float32) or 'grey' (1-channel uint8).
		compression:         HDF5 compression algorithm — 'gzip' or 'lzf'.
	'''
	def __init__(self, root_dir, output_dir, framesize, institution_prefix, channels, compression):
		self.root_dir = root_dir
		self.output_dir = output_dir
		self.framesize = framesize
		self.institution_prefix = institution_prefix
		self.compression = compression
		self.channels = channels

	def dcm_to_array(self, input_file):
		'''
		Read a single DICOM file and convert its pixel data to a numpy array.

		RGB DICOMs are converted to greyscale via luminosity weighting and then
		repeated across 3 channels for compatibility with pretrained RGB models.
		2D (single-frame) DICOMs are expanded to 3-channel format. Extracts
		SeriesDescription, SliceLocation, AccessionNumber, PatientID, and
		InstanceNumber as metadata alongside the pixel array.

		Args:
			input_file: Path to a single .dcm file.

		Returns:
			Tuple of (array [c, h, w], series, frame_loc, accession, mrn, unique_frame_index),
			or None if the DICOM is corrupted or unreadable.
		'''
		try:
			df = dcm.dcmread(input_file)
			# Check if any dicoms have non greyscale 
			df.PhotometricInterpretation = 'MONOCHROME2'

			# Save series name (with some cleanup) + frame location 
			series = df.SeriesDescription.replace(" ","_")
			series = series.replace("/","_")
			frame_loc = df.SliceLocation
			accession = df.AccessionNumber  
			mrn = df.PatientID 

			if len(df.pixel_array.shape) == 3: 
				# f, w, h, c
				frames = df.pixel_array.shape[0]
				r, g, b = df.pixel_array[:,:,:,0], df.pixel_array[:,:,:,1], df.pixel_array[:,:,:,2]
				gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

				# c, f, w, h
				array = np.repeat(gray[None,...],3,axis=0)

			elif len(df.pixel_array.shape) == 2:
				array = np.repeat(df.pixel_array[None,...],3,axis=0)

			else:
				# Placeholder empty frame so things don't break
				print('Invalid dimensions')
		
			try:
				unique_frame_index = f'{df.InstanceNumber}'
			except:
				print('Could not extract InstanceNumber')
				unique_frame_index = None

				
			return array, series, frame_loc, accession, mrn, unique_frame_index

		except Exception as ex:
			print("DICOM corrupted! Skipping...")
			print(ex)


	def collate_arrays(self, dcm_subfolder, stacked=False):
			'''
			Combine a folder (or list of folders) of per-frame DICOMs into a single sorted 4D array.

			Each DICOM file in a series represents one frame. Frames are sorted first by
			SliceLocation then by InstanceNumber using natsort to ensure correct temporal
			and spatial ordering. Slice boundary indices are computed for multi-slice sequences
			(e.g. SAX stacks). Torchvision v2 transforms are applied for resize and center crop.

			In stacked mode (e.g. UK Biobank SAX spread across multiple subfolders), dcm_subfolder
			is a list of folder paths that are iterated and combined before sorting.

			Institution-specific MRN/accession overrides are applied here for medstar and upenn,
			where DICOM metadata fields are blank and identifiers must be parsed from the tar filename.

			Args:
				dcm_subfolder: Path to a single series folder, or a list of paths when stacked=True.
				stacked:       If True, treat dcm_subfolder as a list of folders for multi-folder series.

			Returns:
				Tuple of (collated_array [f, c, h, w], series, slice_frames, total_images, mrn, accession),
				or None if the array cannot be constructed (ragged frames, invalid size, etc.).
			'''
			if stacked:
				total_images = 0
				video_list = []
				slice_location = []
				unique_frame_index = []
				# For ukbiobank SAX the 'dcm_subfolder' is a list of 'dcm_subfolders'
				for folder in dcm_subfolder:
					dcm_list = os.listdir(folder)
					total_images += len(dcm_list)

					# dcm_list is completely unsorted
					for d in dcm_list:
						dcm_data = self.dcm_to_array(os.path.join(folder, d))
						if dcm_data is not None:
							video_list.append(dcm_data[0])
							slice_location.append(dcm_data[2])
							series = dcm_data[1]
							accession = dcm_data[3]
							mrn = dcm_data[4]
							unique_frame_index.append(dcm_data[5])
							
						else:
							continue
			else:
				dcm_list = os.listdir(dcm_subfolder)
				total_images = len(dcm_list)
				video_list = []
				slice_location = []
				unique_frame_index = []

				# dcm_list is completely unsorted
				for d in dcm_list:
					dcm_data = self.dcm_to_array(os.path.join(dcm_subfolder, d))

					if dcm_data is not None:
						video_list.append(dcm_data[0])
						slice_location.append(dcm_data[2])
						series = dcm_data[1]
						accession = dcm_data[3]
						mrn = dcm_data[4]
						unique_frame_index.append(dcm_data[5])
					else:
						continue

			try:
				#NEW
				tmp_df = pd.DataFrame({'unique_frame_index':unique_frame_index, 'slice_location':slice_location})
				tmp_df = tmp_df.sort_values(by=['slice_location','unique_frame_index'], key=natsort_keygen())
				
				reordered_index = tmp_df.index.tolist()
				slice_location = tmp_df['slice_location'].tolist()
				video_list = [video_list[i] for i in reordered_index]

				### Torchvision transformations replacement ###
				collated_array = torch.Tensor(np.array(video_list))
				slice_frames = np.where(np.array(slice_location)[:-1] != np.array(slice_location)[1:])[0]
				transforms = v2.Compose([v2.Resize(size=self.framesize), v2.CenterCrop(round(0.75*self.framesize))])
				collated_array = transforms(collated_array) # returns as [f, c, h, w]
				#collated_array = collated_array.transpose(1, 0) # returns as [c, f, h, w] for now ##TODO: REMOVE AND SWITCH TO STORING GREYSCALE f, c, h, w	 

				'''
				Specific workarounds for strange institution specific data handling
				- UKBIOBANK: 
				  Has spaces in mrn, and accessions have a weird format that needs cleaning
				- MEDSTAR: 
				  dicom data has blank mrn and accessions, that info is pulled directly from tar filename instead
				- UPENN:
				  dicom data is anonymized, mrn and accession is taken from tar filenames
				'''

				if self.institution_prefix == "medstar" or self.institution_prefix == "upenn":
					mrn = self.filename.split('-')[0]
					accession = self.filename.split('-')[1][:-4]

				return collated_array, series, slice_frames, total_images, mrn, accession

			except ValueError as v: 
				print(v)
				print('Ragged numpy arrays, Skipping..')
				collated_array = None
				slice_frames = None
				return None

			except Exception as e:
				print(e)
				print('Invalid array size. Skipping...')
				return None
			

	def array_to_h5(self, collated_array, series, slice_indices, total_images, mrn, accession):
		'''
		Write a collated array to an HDF5 file as a named dataset with metadata attributes.

		Output is written to: output_dir/institution_mrn/accession.h5
		Each series is stored as a separate dataset within the HDF5 file, keyed by
		SeriesDescription. Files are opened in append mode so multiple series from the
		same accession are accumulated into a single .h5 file across separate calls.

		In greyscale mode, the array is globally normalized and cast to uint8 before
		writing, reducing storage by ~50-70% versus float32 RGB. The channel dimension
		is dropped to store as [f, h, w].

		Duplicate series keys are skipped silently (HDF5 dataset already exists).

		Args:
			collated_array: Torch tensor of shape [f, c, h, w] (or [f, h, w] for grey).
			series:         SeriesDescription string used as the HDF5 dataset key.
			slice_indices:  1D array of frame indices where slice location changes (SAX stacks).
			total_images:   Total number of source DICOM frames before collation.
			mrn:            Patient MRN string used to name the output parent directory.
			accession:      Accession number string used as the HDF5 filename.

		Returns:
			Full path to the written HDF5 file.
		'''
		dytpe_setting = 'f'
		if self.channels == "grey":
			## Normalize globally ##
			# This requires dtype to be manually set to "uint8" to truly work and yield storage savings # 
			vmin, vmax = torch.min(collated_array), torch.max(collated_array)
			collated_array = torch.clamp((collated_array - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).to(torch.uint8)
			dytpe_setting = 'uint8'
			
			if len(collated_array.shape) == 4:
				collated_array = collated_array[:,1,:,:]
			elif len(collated_array.shape) == 3:
				collated_array = collated_array[1,:,:]


		os.makedirs(os.path.join(self.output_dir, self.institution_prefix + '_' + mrn), exist_ok=True)
		new_filename = accession + '.h5'
		
		# Create hdf5 file or append to existing if available
		h5f = h5py.File(os.path.join(self.output_dir, self.institution_prefix + '_' + mrn, new_filename), 'a')
		print(f'Exporting {accession}-{series} as hdf5 dataset...')
		
		# Store each series as an array (Skips if series already exists. Might need to rework this 
		try:
			dset = h5f.create_dataset(series, data=collated_array, dtype=dytpe_setting, compression=self.compression)

			# Attributes
			dset.attrs.create('slice_frames', slice_indices, dtype='i')
			dset.attrs.create('total_images', total_images, dtype='i')

		except ValueError:
			print(f'{series} already exists. Skipping...')
			pass

		except RuntimeError:
			print(f'{series} already exists. Skipping...')
			pass

		h5f.close()
		return os.path.join(os.path.join(self.output_dir, self.institution_prefix + '_' + mrn, new_filename))


	def view_disambugator(self, dcm_directory):
		'''
		Group DICOM subfolders by SeriesDescription and route each group through collation.

		Iterates all subdirectories in a extracted tar archive, reads the SeriesDescription
		from the first DICOM in each folder, and builds a map of series → [folder, ...].
		Series split across multiple folders (e.g. UK Biobank SAX stacks) are collated in
		stacked mode after sorting folders by SliceLocation. Single-folder series are
		collated normally. InlineVF overlay series and folders with >2500 frames are skipped.
		Multi-folder series are trimmed to a maximum of 10 slices (500 frames) to prevent
		memory blowups.

		Args:
			dcm_directory: List of subdirectory paths from glob expansion of the extracted tar.

		Returns:
			Path to the last HDF5 file written (used for optional GCS upload queue).
		'''

		series_map = defaultdict(list)

		for dcm_subfolder in dcm_directory:
			files = glob.glob(os.path.join(dcm_subfolder, "*"))
			if not files:
				continue
			
			if len(files) < 2500:
				try:
					df = dcm.dcmread(files[0], stop_before_pixels=True)
					series = df.SeriesDescription

					if "InlineVF" in series:
						print(f"Skipping InlineVF overlay...")
						continue
	 
					series_map[series].append(dcm_subfolder)

				except Exception as e:
					print(f"Failed to parse DICOM in {dcm_subfolder}: {e}")
					continue

			else: 
				print(f"Insane number of frames detected: {len(files)}; skipping...")
				continue

		# upenn_sax_folder_list = [] ### remove line later
		for series, folders in series_map.items():
			# Sort folders by df.SliceLocation if multiple separate folders present
			if len(folders) > 1:
				print(f"Processing {series} across {len(folders)} folders ...")	
				try:
					folders.sort(
						key=lambda x: dcm.dcmread(os.path.join(x, os.listdir(x)[0])).SliceLocation,
						reverse=True,
					)
				except Exception:
					pass  

				#OpenCV limitation where resize operation can only take place on less than 512 frames at a time, hard code to maximum of 10 slices (500 frames)
				if len(folders) > 10:
					print("Trimming to first 10 slices")
					folders = folders[:10]

				# Guard against OOM: per-folder cap is 2500 but stacked series multiply that.
				# Trim folders until total frame count fits within 2500.
				total_frames = sum(len(os.listdir(f)) for f in folders)
				while total_frames > 2500 and len(folders) > 1:
					folders = folders[:-1]
					total_frames = sum(len(os.listdir(f)) for f in folders)
				if total_frames > 2500:
					print(f"Stacked series {series} exceeds 2500 frames even as single folder, skipping...")
					continue
				print(f"Stacked series {series}: {len(folders)} folders, {total_frames} total frames")

				collated_array = self.collate_arrays(folders, stacked=True)
				if collated_array is not None:
					h5_path = self.array_to_h5(*(collated_array))

			else:
				collated_array = self.collate_arrays(folders[0], stacked=False)
				if collated_array is not None:
					h5_path = self.array_to_h5(*(collated_array))

		return h5_path


	def process_dicoms(self, filename, queue=None):
		'''
		Top-level processing function for a single tar.gz DICOM archive.

		Extracts the archive to TMP_DIR, runs view_disambugator() to convert all
		series to HDF5, then removes the extracted directory to reclaim disk space.
		Disk usage is throttled before extraction when a GCS upload queue is active
		(via wait_if_disk_full). The resulting HDF5 path is pushed to the queue for
		asynchronous GCS upload if provided.

		Designed to be called via multiprocessing.Pool.apply_async() for parallel
		processing across many tar files.

		Args:
			filename: Basename of the .tgz file within root_dir.
			queue:    Optional multiprocessing.Queue for GCP_Upload_Manager. If None,
			          files are written locally only and no throttling is applied.
		'''
		self.filename = filename
		
		if queue is not None:
			### Throttle function if disk sage > 90% ###
			wait_if_disk_full(TMP_DIR)
		
		tar = tarfile.open(os.path.join(self.root_dir, filename))
		tar_extract_path = os.path.join(TMP_DIR, filename[:-4])
		tar.extractall(tar_extract_path)
		tar.close()

		# List series folders and iterate over them all one by one 
		# Return arrays for each folder, convert to hdf5 therafter
		print(f'Extracted tarfile for {self.filename[:-4]} ...')
		dcm_directory = glob.glob(os.path.join(tar_extract_path, '*', '*'))

		# Handles separate pipelines based on data source
		h5_path = self.view_disambugator(dcm_directory)

		# Clean up after to save space  
		try:
			rmtree(tar_extract_path, ignore_errors=True)

		except Exception as ex:
			print('Failed to purge TMP_DIR(s)')
		
		print(f'Completed processing {self.filename}')

		if queue is not None:
			queue.put(h5_path)

if __name__ == '__main__':

	parser = ap.ArgumentParser(
		description="Preprocess dicom to hdf5 v2.0",
		epilog="Version 2.0; Created by Rohan Shad, MD"
	)

	parser.add_argument('-r', '--root_dir', metavar='', required=False, help='Full path to root directory OR bucket GCP gs:bucket_name', default='/Users/rohanshad/PHI Safe/test_mri_downloads')
	parser.add_argument('-l', '--csv_list', metavar='', required=False, help='Process only files listed in csv_list.csv', default=None)
	parser.add_argument('-o', '--output_dir', metavar='', required=True, help='Path to output directory')
	parser.add_argument('-z', '--compression', metavar='', required=False, help='Compression type (gzip pr lzf)', default='gzip')
	parser.add_argument('-c', '--cpus', metavar='', type=int, default='4',help='number of cores to use in multiprocessing')
	parser.add_argument('-d', '--debug', action='store_true', default=False)
	parser.add_argument('-s', '--framesize', metavar='', type=int, default='480', help='framesize in pixels')
	parser.add_argument('-v', '--visualize', action='store_true', required=False, help='print data from random hdf5 file in output folder')
	parser.add_argument('-i', '--institution', metavar='', required=True, help='institution name to use as prefix for hdf5 files')
	parser.add_argument('--gcs_bucket_upload', metavar='', default=None, help='gs:bucket destination for files to be directly uploaded to from local tmp_output directory (-o)')
	parser.add_argument('--channels', metavar='', default="rgb", help='Saves hdf5 array either as 3 channel "rgb" or 1 channel "grey" to optimize storage space')

	args = vars(parser.parse_args())
	print(args)

	root_dir = args['root_dir']
	csv_list = args['csv_list']
	compression = args['compression']
	cpus = args['cpus']
	visualize = args['visualize']
	institution_prefix = args['institution']
	framesize = args['framesize']
	debug = args['debug']
	gcs_bucket_upload = args["gcs_bucket_upload"]
	channels = args["channels"]
	if gcs_bucket_upload is not None:
		assert gcs_bucket_upload[:3] == "gs:"

	#For gcloud:
	output_dir = args['output_dir']
	os.makedirs(output_dir, exist_ok=True)	

	#### Debugging lines ####
	if debug == True:
		print('Running in debug mode...')
		if csv_list is not None:
			df = pd.read_csv(os.path.join(root_dir, csv_list))
			print(df)
			filenames = df['filenames'].tolist()
			files_in_dir = os.listdir(root_dir)
			filenames = set(filenames).intersection(files_in_dir)

		else:
			filenames = os.listdir(root_dir)
			
		print('------------------------------------')
		print(f'Total scans (Accession numbers) available: {len(filenames)}')
		print('Checking failure rate..')

		outputs = os.listdir(output_dir)
		processed_files = []
		for o in outputs:
			
			if o.partition('_')[0] == institution_prefix:
				institution, sep, mrn = o.partition('_')
				acc = glob.glob(os.path.join(output_dir,o,'*'))
				for item in acc:
					filename = mrn + '-' + os.path.basename(item[:-2]) + 'tgz'
					
					# Hack for ukbiobank:
					if institution == 'ukbiobank':
						filename = mrn+'.tgz'

					processed_files.append(filename)

		incomplete = set(filenames).difference(set(processed_files))
		print('Successfully processed accessions:', len(processed_files))
		print(f'Did not process {len(incomplete)} files.') 
		print('Exporting failed runs...')
		print('------------------------------------')

		incomplete_df = pd.DataFrame(list(incomplete), columns = ["filenames"])
		incomplete_df.to_csv(os.path.join(output_dir,'failed_to_process.csv'), index=False)

	#### Main DCM to HDF5 conversion pipeline ####
	else:
		# Main run command to convert dcm files to hdf5
		p = multiprocessing.Pool(processes=cpus)

		if root_dir[:3] == "gs:":
			# Split / to ensure mount point doesn't duplicate subdirs if present
			if mount_gcs_bucket(root_dir, f'{TMP_DIR}/mnt/{root_dir[3:].split("/")[0]}') is True:
				root_dir = f'{TMP_DIR}/mnt/{root_dir[3:]}'
		else:
			pass

		if csv_list is not None:
			try:
				df = pd.read_csv(os.path.join(root_dir, csv_list))
				print(df)
				filenames = df['filenames'].tolist()

				files_in_dir = os.listdir(root_dir)
				filenames = set(filenames).intersection(files_in_dir)
			except:
				print('Could not open csv safelist')
		else:
			filenames = os.listdir(root_dir)

		start_time = time.time()
		mri_processor = CMRI_PreProcessor(root_dir, output_dir, framesize, institution_prefix, channels, compression)
		if gcs_bucket_upload is not None:
			try:
				manager = multiprocessing.Manager()
				shared_queue = manager.Queue()
				gcs_manager = GCP_Upload_Manager(output_dir, shared_queue, gcs_bucket_upload[3:])
				gcs_manager.start()

			except Exception as ex:
				print(ex)
		else:
			shared_queue = None

		async_results = {}
		for f in filenames:
			# Only loops through tgz files
			if f[-3:] == 'tgz':
				if cpus > 1:
					async_results[f] = p.apply_async(mri_processor.process_dicoms, [f, shared_queue])
				else:
					mri_processor.process_dicoms(f, shared_queue)

			else:
				print("No tar files here!")
				continue

		p.close()

		# Collect results with per-task timeout so a single hung worker
		# (e.g. blocked tarfile.extractall or rmtree on a bad mount) cannot stall the pool.
		TASK_TIMEOUT = 1000  # seconds — adjust if legitimate scans take longer
		timed_out = []
		failed = []
		for f, result in async_results.items():
			try:
				result.get(timeout=TASK_TIMEOUT)
			except multiprocessing.TimeoutError:
				print(f'WARN: {f} timed out after {TASK_TIMEOUT}s — skipping')
				timed_out.append(f)
			except Exception as ex:
				print(f'WARN: {f} raised an exception — skipping')
				print(ex)
				failed.append((f, str(ex)))

		# Workers that timed out are still alive and stuck — terminate the pool
		# before joining, otherwise p.join() hangs waiting for them to exit.
		p.terminate()
		p.join()

		# Clean up tmp dirs left by terminated workers (they never ran rmtree).
		# Done from main process after workers are dead so any held file locks are released.
		for f in timed_out:
			leftover = os.path.join(TMP_DIR, f[:-4])
			if os.path.exists(leftover):
				print(f'Cleaning up leftover tmp dir for {f}...')
				rmtree(leftover, ignore_errors=True)

		# Write stalled/failed runs to a log file for post-hoc review.
		if timed_out or failed:
			run_date = time.strftime('%Y-%m-%d')
			log_path = os.path.join(output_dir, f'{institution_prefix}_{run_date}_stalledruns.log')
			with open(log_path, 'a') as log:
				log.write(f'# Run started {time.strftime("%Y-%m-%d %H:%M:%S")} — institution: {institution_prefix}\n')
				for f in timed_out:
					log.write(f'TIMEOUT\t{f}\n')
				for f, reason in failed:
					log.write(f'EXCEPTION\t{f}\t{reason}\n')
			print(f'Stalled/failed runs logged to {log_path}')

		if shared_queue:
			shared_queue.put(None)
			gcs_manager.wait_until_done()

		try:
			unmount_gcs_bucket(root_dir, f'{TMP_DIR}/mnt/{root_dir[3:]}')
		except:
			pass

		elapsed = round((time.time() - start_time), 2)
		print('------------------------------------')
		print(f'Elapsed time: {elapsed}s')
		print(f'Timed out:    {len(timed_out)} scan(s)')
		print(f'Failed:       {len(failed)} scan(s)')
		if timed_out or failed:
			print(f'See {institution_prefix}_{time.strftime("%Y-%m-%d")}_stalledruns.log for details')
		print('------------------------------------')

		# Notification via Slack
		notify_slack(f"preprocess_mri.py job status: complete. \nTotal time: {elapsed}s\nTimed out: {len(timed_out)} | Failed: {len(failed)}")


