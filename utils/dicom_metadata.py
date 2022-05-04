'''
Pulls and plots dicom metadata from dcm folders
Use this to generate a list of 
'''

import torch
import matplotlib
import matplotlib.pyplot as plt
import random
import os 
import numpy as np
import argparse as ap
import glob
import pydicom as dcm
import time
import multiprocessing
import tarfile
from shutil import rmtree
import pandas as pd
from collections import Counter
from pyaml_env import BaseConfig, parse_config


# Change these variables for different user / cluster
cfg = BaseConfig(parse_config('../local_config.yaml'))
TMP_DIR = cfg.tmp_dir

def dcm_reader(dcm_subfolder):
	'''
	Reads in dicom file and reads the metadata
	'''
	dicom_list = os.listdir(dcm_subfolder)
	
	try:
		df = dcm.dcmread(os.path.join(dcm_subfolder, dicom_list[1]))
		# Check if any dicoms have non greyscale 
		df.PhotometricInterpretation = 'MONOCHROME2'

		# Save series name + frame location 
		series = df.SeriesDescription.replace(" ","_")
		series = series.replace("/","_")
		
		frame_loc = df.SliceLocation
		accession = df.AccessionNumber  
		mrn = df.PatientID 

		return series, frame_loc, mrn+'-'+accession

	except Exception as ex:
		print("DICOM corrupted! Skipping...")
		print(ex)


def process_dicoms(root_dir, filename):
	tar = tarfile.open(os.path.join(root_dir, filename))
	tar_extract_path = os.path.join(TMP_DIR, filename[:-4])
	tar.extractall(tar_extract_path)
	tar.close()

	# List series folders and iterate over them all one by one 

	print("Extracted tarfile for", filename[:-4], "...")

	dcm_directory = glob.glob(os.path.join(tar_extract_path, '*', '*'))

	metadata_minilist = []
	for dcm_subfolder in dcm_directory:
		if len(glob.glob(os.path.join(dcm_subfolder, '*'))) > 1:
			metadata_minilist.append(dcm_reader(dcm_subfolder))
		else:
			print('Skipped single image series')

	# Clean up after to save space	
	try:
		rmtree(tar_extract_path, ignore_errors=True)
	except Exception as ex:
		print('Failed to purge TMP_DIR')
	
	print('Completed processing', filename)
	
	return metadata_minilist

def safe_makedir(path):
	'''
	Nick's little safe mkdir function
	'''
	if not os.path.exists(path):
		os.makedirs(path)



if __name__ == '__main__':

	parser = ap.ArgumentParser(
		description="Reads dicom file and pulls metadata",
		epilog="Version 0.1; Created by Rohan Shad, MD"
	)

	parser.add_argument('-r', '--root_dir', metavar='', required=False, help='Full path to root directory', default='/Users/rohanshad/PHI Safe/test_mri_downloads/archive')
	parser.add_argument('-l', '--csv_list', metavar='', required=False, help='Process only files listed in csv_list.csv', default=None)
	parser.add_argument('-o', '--output_dir', metavar='', required=False, help='Where all output files will be stored')
	parser.add_argument('-c', '--cpus', metavar='', type=int, default='4',help='number of cores to use in multiprocessing')
	parser.add_argument('-s', '--framesize', metavar='', type=int, default='480', help='framesize in pixels')
	parser.add_argument('-v', '--visualize', action='store_true', required=False, help='print data from random hdf5 file in output folder')
	
	args = vars(parser.parse_args())
	print(args)

	root_dir = args['root_dir']
	csv_list = args['csv_list']
	cpus = args['cpus']
	visualize = args['visualize']
	framesize = args['framesize']

	#For gcloud:
	output_dir = args['output_dir']
	safe_makedir(output_dir)	


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

	async_results = []
	start_time = time.time()
	for f in filenames:
		# Only loops through tgz files
		if f[-3:] == 'tgz':
			if cpus > 1:
				async_results.append(p.apply_async(process_dicoms, [root_dir, f]))
				#print(async_result.get())
				
			else:
				async_results.append(process_dicoms(root_dir, f))

		else:
			#print("Not a DICOM file")
			continue


	p.close()
	p.join()

	print('------------------------------------')
	print('Collecting results...')
	final_list = []
	for i in async_results:
		sublist = i.get()
		for item in sublist:
			final_list.append(item)

	# Export csv file containing view, frame_loc (or any random metadata element here), and filename (single accession number)
	metadata_df = pd.DataFrame(final_list, columns=['series_description','frame_loc','filename'])
	print(metadata_df)
	freq = metadata_df['series_description'].value_counts().reset_index()
	freq.columns = ['series_description', 'frequency']
	print(freq)
	freq.to_csv('metadata_freq.csv', index=False)
	metadata_df.to_csv('metadata.csv', index=False)

	print()
	print(f'Saved metadata from {len(async_results)} files (Accession numbers)')
	print(f'Total number of views: {len(final_list)}')
	print(f'Elapsed time: {round((time.time() - start_time), 2)}')
	print('------------------------------------')

