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
from natsort import natsorted

from pyaml_env import BaseConfig, parse_config

# Read local_config.yaml for local variables 
cfg = BaseConfig(parse_config('../local_config.yaml'))
TMP_DIR = cfg.tmp_dir


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
			df = dcm.dcmread(input_file, force=True)

			# Check if any dicoms have non greyscale 
			df.PhotometricInterpretation = 'MONOCHROME2'

			# Save series name (with some cleanup) + frame location 
			series = df.SeriesDescription.replace(" ","_")
			series = series.replace("/","_")
			frame_loc = df.SliceLocation
			accession = df.AccessionNumber  
			mrn = df.PatientID 

			# TODO:
			### len==3 section probably buggy af since at this point its all images not video ###
			if len(df.pixel_array.shape) == 3: 
				# f, w, h, c
				frames = df.pixel_array.shape[0]
				r, g, b = df.pixel_array[:,:,:,0], df.pixel_array[:,:,:,1], df.pixel_array[:,:,:,2]
				gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

				# c, f, w, h
				array = np.repeat(gray[None,...],self.channels,axis=0)

			elif len(df.pixel_array.shape) == 2:
				array = np.repeat(df.pixel_array[None,...],self.channels,axis=0)
				# array here is c, f, w, h

			else:
				# Placeholder empty frame so things don't break
				print('Invalid dimensions')

			return array, series, frame_loc, accession, mrn

		except Exception as ex:
			print("DICOM corrupted! Skipping...")
			print(ex)


	def collate_arrays(self, dcm_subfolder, sax_stacked=False):
		'''
		MRI sequences save each frame as a separate array with it's own metadata. 
		Function collate_arrays combines each folder of dcm images into a single array
		Returns reorderd video array, series name, slice frames for multi-slice sequences, total images
		'''
		if sax_stacked:
			total_images = 0
			video_list = []
			slice_location = []
			# For ukbiobank SAX the 'dcm_subfolder' is a list of 'dcm_subfolders'
			for folder in dcm_subfolder:
				dcm_list = os.listdir(folder)
				try:
					dcm_list.natsorted()
				except:
					print('WARNING: Potential sorting error for SAX files')
					dcm_list.sort()
				total_images += len(dcm_list)
				for d in dcm_list:
					dcm_data = self.dcm_to_array(os.path.join(folder, d))
					if dcm_data is not None:
						video_list.append(dcm_data[0])
						slice_location.append(dcm_data[2])
						series = dcm_data[1]
						mrn = dcm_data[4]
					else:
						continue
		else:
			dcm_list = os.listdir(dcm_subfolder)
			try:
				dcm_list = natsorted(dcm_list)
			except:
				print('WARNING: Potential sorting error for SAX files')
				dcm_list.sort()
			total_images = len(dcm_list)
			video_list = []
			slice_location = []
			# dcm_list has been sorted by int key if dcm files are simple numbers (essential otherwise it sorts weird as a string)
			for d in dcm_list:
				dcm_data = self.dcm_to_array(os.path.join(dcm_subfolder, d))
				if dcm_data is not None:
					video_list.append(dcm_data[0])
					slice_location.append(dcm_data[2])
					series = dcm_data[1]
					accession = dcm_data[3]
					mrn = dcm_data[4]
				else:
					continue

		try:
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
			'''
			if self.institution_prefix == "ukbiobank":
				accession = mrn.replace(" ", "")
				mrn = dcm_subfolder.split('/')[-2]
			
			if self.institution_prefix == "medstar":
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
			dset.attrs.create('slice_frames', slice_indices, dtype='i')
			dset.attrs.create('total_images', total_images, dtype='i')

		except ValueError:
			print(f'{series} already exists. Skipping...')
			pass

		except RuntimeError:
			print(f'{series} already exists. Skipping...')
			pass

		h5f.close()


	def ukbiobank_mri_pipeline(self, dcm_directory):
		'''
		Handles specific nuances of ukbiobank data
		'''
		sax_files_list = []
		for dcm_subfolder in os.listdir(dcm_directory):
			dicom_list = os.listdir(dcm_subfolder)
			df = dcm.dcmread(os.path.join(dcm_subfolder, dicom_list[0]), force=True)
			
			if "CINE_segmented_SAX" in df.SeriesDescription and "InlineVF" not in df.SeriesDescription:
				sax_files_list.append(dcm_subfolder)
				print(f'Stacking {df.SeriesDescription}')

			# Process non SAX ukbiobank studies
			else:
				if "InlineVF" in df.SeriesDescription:
					print('Skipping scan with random overlay...')
				else:
					if len(glob.glob(os.path.join(dcm_subfolder, '*'))) > 1:
						collated_array = self.collate_arrays(dcm_subfolder, sax_stacked=False)  

					if collated_array is not None:
						self.array_to_h5(*(collated_array))

		# Process ukbiobank SAX subfolders all as a single stack after reordering 
		if len(sax_files_list) > 0:	
			sax_files_list.sort(key=lambda x: dcm.dcmread(os.path.join(x,os.listdir(x)[0])).SliceLocation, reverse=True)
			
			# opencv limitation where resize operation can only take place on less than 512 frames at a time, hard code to maximum of 10 slices (500 frames)
			print(f'{len(sax_files_list[0:10])} SAX slices to export...')
			collated_array = self.collate_arrays(sax_files_list[0:10], sax_stacked=True)

			if collated_array is not None:
				self.array_to_h5(*(collated_array))

	def view_disambugator(self, dcm_directory):
		'''
		Iterate over multiple subfolders inside dcm_directory for all non-ukbiobank datasets
		'''
		# Process separated views

		sax_files_list = []
		for dcm_subfolder in glob.glob(os.path.join(dcm_directory, '*')):
			dicom_list = os.listdir(dcm_subfolder)
			df = dcm.dcmread(os.path.join(dcm_subfolder, dicom_list[0]), force=True)

			# MEDSTAR
			if self.institution_prefix == 'medstar':
				if df.SeriesDescription in ["tfisp_cine_sax","short_axis_cine","short_axis_cine_trufi_retro", "short_axis_stack_cine_trufi_retro"]:
					if len(df.pixel_array.shape) == 2:
						sax_files_list.append(dcm_subfolder)
					print(f'Stacking {df.SeriesDescription}')

				else:
					if len(glob.glob(os.path.join(dcm_subfolder, '*'))) > 1:
						collated_array = self.collate_arrays(dcm_subfolder, sax_stacked=False)  

						if collated_array is not None:
							self.array_to_h5(*(collated_array))
					else:
						print('Skipped single image series')

			# UKBIOBANK
			if self.institution_prefix == 'ukbiobank':
				if df.SeriesDescription in "CINE_segmented_SAX" and df.SeriesDescription not in "InlineVF":
					sax_files_list.append(dcm_subfolder)
					print(f'Stacking {df.SeriesDescription}')

				else:
					if "InlineVF" in df.SeriesDescription:
						print('Skipping scan with random overlay...')
					else:
						if len(glob.glob(os.path.join(dcm_subfolder, '*'))) > 1:
							collated_array = self.collate_arrays(dcm_subfolder, sax_stacked=False)  

							if collated_array is not None:
								self.array_to_h5(*(collated_array))
						else:
							print('Skipped single image series')

		if len(sax_files_list) > 0:
			sax_files_list.sort(key=lambda x: dcm.dcmread(os.path.join(x,os.listdir(x)[0])).SliceLocation, reverse=True)
		
			# opencv limitation where resize operation can only take place on less than 512 frames at a time, hard code to maximum of 10 slices (500 frames)
			print(f'{len(sax_files_list[0:10])} SAX slices to export...')
			collated_array = self.collate_arrays(sax_files_list[0:10], sax_stacked=True)

			if collated_array is not None:
				self.array_to_h5(*(collated_array))

		# OTHER HOSPITALS
		else:
			for dcm_subfolder in glob.glob(os.path.join(dcm_directory, '*')):
				if len(glob.glob(os.path.join(dcm_subfolder, '*'))) > 1:
					collated_array = self.collate_arrays(dcm_subfolder) 
					
					if collated_array is not None:
						self.array_to_h5(*(collated_array))
					else:
						continue
				else:
					print('Skipped single image series')


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
		print(dcm_directory[0])
		self.view_disambugator(dcm_directory[0])

		# Clean up after to save space  
		try:
			rmtree(tar_extract_path, ignore_errors=True)
		except Exception as ex:
			print('Failed to purge TMP_DIR')
		
		print(f'Completed processing {self.filename}')



if __name__ == '__main__':

	parser = ap.ArgumentParser(
		description="Preprocess dicom to hdf5 v0.1",
		epilog="Version 0.1; Created by Rohan Shad, MD"
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

		print(f'Elapsed time: {round((time.time() - start_time), 2)}')

