'''
First Cardiac MRI preprocessing script that reads dicom directories and stores files as shown below:

stanford_RF3da2ty4 (MRN parent folder)
├── RFasfe3581.h5 (Accession Number hdf5 file)
├── RFxlv2173.h5
	├── 4CH_FIESTA_BH 			{data: 4d array} {attr: fps, total images}
	├── SAX_FIESTA_BH_1			{data: 4d array} {attr: fps, total images, slice frame index}
	├── SAX_FIESTA_BH_2			{data: 4d array} {attr: fps, total images, slice frame index}
	├── STACK_LV3CH_FIESTA_BH 	{data: 4d array} {attr: fps, total images, slice frame index}
	
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
from torchvideotransforms import video_transforms, volume_transforms 
import matplotlib.pyplot as plt
import time
from natsort import natsorted, natsort_keygen
import platform
import pylibjpeg
from pyaml_env import BaseConfig, parse_config
from collections import defaultdict

# Read local_config.yaml for local variables 

device = platform.uname().node.replace('-','_')
cfg = BaseConfig(parse_config(os.path.join('..', 'local_config.yaml')))

if 'sh' in device:
	device = 'sherlock'
elif '211' in device:
	device = 'cubic'
TMP_DIR =  getattr(cfg, device).tmp_dir
BUCKET_NAME =  cfg.global_settings.bucket_name


class CMRI_PreProcessor:
	'''
	Process dicoms / tar decompress >> collated_array >> array_to_h5
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
		Reads in dicom file and converts to numpy array
		Greyscale imaging is stored in 3 channels for downstream compatibility with pre-trained models
		Returns: array [c, f, w, h] 
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
				array = np.repeat(gray[None,...],self.channels,axis=0)

			elif len(df.pixel_array.shape) == 2:
				array = np.repeat(df.pixel_array[None,...],self.channels,axis=0)

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
		MRI sequences save each frame as a separate array with it's own metadata. 
		Function collate_arrays combines each folder of dcm images into a single array
		Returns reorderd video array, series name, slice frames for multi-slice sequences, total images
		Uses InstanceNumber to figure out order of frames in videos / sequences
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


			collated_array = np.array(video_list)
			slice_frames = np.where(np.array(slice_location)[:-1] != np.array(slice_location)[1:])[0]
			collated_array = collated_array.transpose(1 , 2 , 3, 0)
			video_transform_list = [video_transforms.Resize(self.framesize), video_transforms.CenterCrop(round(0.75*self.framesize))]
			transforms = video_transforms.Compose(video_transform_list)
			collated_array = transforms(collated_array)
			
			# converts [c, h, w, f] to [c, f, h, w] for pytorchvideo transforms downstream
			collated_array = np.array(collated_array).transpose(0, 3, 1, 2)

			'''
			Specific workarounds for strange institution specific data handling
			- UKBIOBANK: 
			  Has spaces in mrn, and accessions have a weird format that needs cleaning
			- MEDSTAR: 
			  dicom data has blank mrn and accessions, that info is pulled directly from tar filename instead
			- UPENN:
			  dicom data is not-anonymized, mrn and accession is taken from anonymized filenames instead
			'''


			if self.institution_prefix == "ukbiobank":
				accession = mrn.replace(" ", "")
				mrn = dcm_subfolder.split('/')[-2]

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
		Takes in numpy array and converts into h5 file.
		h5 filename is set to parent array filename (accession number etc)
		h5 contains uniquely named datasets for pixel data array from imaging view / modality 
		
		The first few arguments are positional (up to accession)
		'''

		os.makedirs(os.path.join(self.output_dir, self.institution_prefix + '_' + mrn), exist_ok=True)
		new_filename = accession + '.h5'
		
		# Create hdf5 file or append to existing if available
		h5f = h5py.File(os.path.join(self.output_dir, self.institution_prefix + '_' + mrn, new_filename), 'a')
		print(f'Exporting {accession}-{series} as hdf5 dataset...')
		
		# Store each series as an array (Skips if series already exists. Might need to rework this 
		try:
			dset = h5f.create_dataset(series, data=collated_array, dtype='f', compression=self.compression)

			# Attributes
			#dset.attrs.create('fps', fps, dtype='i')
			dset.attrs.create('slice_frames', slice_indices, dtype='i')
			dset.attrs.create('total_images', total_images, dtype='i')

		except ValueError:
			print(f'{series} already exists. Skipping...')
			pass

		except RuntimeError:
			print(f'{series} already exists. Skipping...')
			pass

		h5f.close()


	def view_disambugator(self, dcm_directory):
		'''
		Iterate through multiple subfolders in dcm_directory and group DICOMs by SeriesDescription.
		Handles cases where multiples of same SeriesDescriptions are split across folders.
		'''

		series_map = defaultdict(list)

		for dcm_subfolder in dcm_directory:
			files = glob.glob(os.path.join(dcm_subfolder, "*"))
			if not files:
				continue
			
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

		for series, folders in series_map.items():
			# if self.institution_prefix == 'ukbiobank' and series_desc != "CINE_segmented_SAX":
			# 	continue  # UKBB-specific filter
			
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

				collated_array = self.collate_arrays(folders, stacked=True)
				if collated_array is not None:
					self.array_to_h5(*(collated_array))

			else:
				collated_array = self.collate_arrays(folders[0], stacked=False)
				if collated_array is not None:
					self.array_to_h5(*(collated_array))

	def process_dicoms(self, filename):
		'''
		Parent function to create h5 arrays
		Opens tar directory and reads each folder
		'''
		self.filename = filename

		tar = tarfile.open(os.path.join(self.root_dir, self.filename))
		tar_extract_path = os.path.join(TMP_DIR, filename[:-4])
		tar.extractall(tar_extract_path)
		tar.close()

		# List series folders and iterate over them all one by one 
		# Return arrays for each folder, convert to hdf5 therafter
		print(f'Extracted tarfile for {self.filename[:-4]} ...')
		dcm_directory = glob.glob(os.path.join(tar_extract_path, '*', '*'))

		# Handles separate pipelines based on data source
		self.view_disambugator(dcm_directory)

		# Clean up after to save space  
		try:
			rmtree(tar_extract_path, ignore_errors=True)
		except Exception as ex:
			print('Failed to purge TMP_DIR')
		
		print(f'Completed processing {self.filename}')



if __name__ == '__main__':

	parser = ap.ArgumentParser(
		description="Preprocess dicom to hdf5 v0.1",
		epilog="Version 2.0; Created by Rohan Shad, MD"
	)

	parser.add_argument('-r', '--root_dir', metavar='', required=False, help='Full path to root directory', default='/Users/rohanshad/PHI Safe/test_mri_downloads')
	parser.add_argument('-l', '--csv_list', metavar='', required=False, help='Process only files listed in csv_list.csv', default=None)
	parser.add_argument('-o', '--output_dir', metavar='', required=True, help='Path to output directory')
	parser.add_argument('-z', '--compression', metavar='', required=False, help='Compression type (gzip pr lzf)', default='gzip')
	parser.add_argument('-c', '--cpus', metavar='', type=int, default='4',help='number of cores to use in multiprocessing')
	parser.add_argument('-d', '--debug', action='store_true', default=False)
	parser.add_argument('-s', '--framesize', metavar='', type=int, default='480', help='framesize in pixels')
	parser.add_argument('-v', '--visualize', action='store_true', required=False, help='print data from random hdf5 file in output folder')
	parser.add_argument('-i', '--institution', metavar='', required=True, help='institution name to use as prefix for hdf5 files')
	
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

	#For gcloud:
	output_dir = args['output_dir']
	os.makedirs(output_dir, exist_ok=True)	

	#### Visualize one frame from hdf5 MRI array ####
	if visualize == True:
		filenames = glob.glob(os.path.join(output_dir,'*','*'))
		file_list_final = []
		
		for f in filenames:
			if ".h5" in f:
				file_list_final.append(f)
			
		if file_list_final == []:
			print('No hdf5 files found..')

		else:
			random_file = random.choice(file_list_final)
			start_time = time.time()
			
			dat = h5py.File(os.path.join(output_dir, random_file), 'r')
			dat.visit(print)

			#Reading hdf5 file once is faster when you have to open multiple arrays from it afterwards (I/O bound)
			dat = dat.get(random.choice(list(dat.keys())))
			print(time.time() - start_time)
			
			print(dat)
			
			# Plotting Code (hdf5 is saved as [c, f h, w])
			array = np.array(dat).transpose(1, 2, 3, 0)

			#time = np.size(array, 0) / dat.attrs['fps']
			#subsample_rate = dat.attrs['fps'] // 20
			print(random_file)
			print(f'number of frames: {np.size(array, 0)}')
			plt.imshow((array[random.choice(list(range(np.size(array, 0))))])[:,:,1]/255, cmap='gist_gray')
			plt.show()

	#### Debugger Module ####
	elif debug == True:
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

	#### Main DCM to HDF5 conversion module ####
	else:
		# Main run command to convert dcm files to hdf5
		p = multiprocessing.Pool(processes=cpus)

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
		mri_processor = CMRI_PreProcessor(root_dir, output_dir, framesize, institution_prefix, 3, compression)

		for f in filenames:
			# Only loops through tgz files
			if f[-3:] == 'tgz':
				if cpus > 1:
					p.apply_async(mri_processor.process_dicoms, [f])
				else:
					mri_processor.process_dicoms(f)

			else:
				print("No tar files here!")
				continue

		p.close()
		p.join()

		print('------------------------------------')
		print(f'Elapsed time: {round((time.time() - start_time), 2)}s')
		print('------------------------------------')

