'''
Pulls and plots dicom metadata from dcm folders
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
import platform
from pyaml_env import BaseConfig, parse_config
from pathlib import Path


# Change these variables for different user / cluster
device = platform.uname().node.replace('-','_')
cfg = BaseConfig(parse_config(Path(__file__).parent.resolve().parent.joinpath('local_config.yaml')))
if 'sh' in device:
	device = 'sherlock'
elif '211' in device:
	device = 'cubic'
ATTN_DIR = getattr(cfg, device).attn_dir
TMP_DIR = getattr(cfg, device).tmp_dir


class Dicom_Metadata_Scanner:
	'''
	Pulls metadata from original dicom scans stored in tar archives
	'''
	def __init__(self, root_dir, output_dir, institution_prefix):
		self.root_dir = root_dir
		self.output_dir = output_dir
		self.institution_prefix = institution_prefix

	def dcm_reader(self, dcm_subfolder):
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
			scanner = df.Manufacturer
			#scanner = df.ManufacturerModelName
			field_strength = df.MagneticFieldStrength

			if institution_prefix == 'medstar':
				mrn = self.filename.split('-')[0]
				accession = self.filename.split('-')[1][:-4]

			if institution_prefix == "ukbiobank":
				accession = mrn.replace(" ", "")
				mrn = dcm_subfolder.split('/')[-2]

			return series, frame_loc, scanner, field_strength, self.institution_prefix+'_'+mrn, accession

		except Exception as ex:
			print("DICOM corrupted! Skipping...")
			print(ex)


	def process_dicoms(self, filename):
		self.filename = filename
		tar = tarfile.open(os.path.join(self.root_dir, self.filename))
		tar_extract_path = os.path.join(TMP_DIR, self.filename[:-4])
		tar.extractall(tar_extract_path)
		tar.close()

		# List series folders and iterate over them all one by one 

		print("Extracted tarfile for", self.filename[:-4], "...")

		dcm_directory = glob.glob(os.path.join(tar_extract_path, '*', '*'))

		metadata_minilist = []
		for dcm_subfolder in dcm_directory:
			if len(glob.glob(os.path.join(glob.escape(dcm_subfolder), '*'))) > 1:
				metadata_minilist.append(self.dcm_reader(dcm_subfolder))
			else:
				print('Skipped single image series')

		# Clean up after to save space	
		try:
			rmtree(tar_extract_path, ignore_errors=True)
		except Exception as ex:
			print('Failed to purge TMP_DIR')
		
		print('Completed processing', filename)
		
		return metadata_minilist



if __name__ == '__main__':

	parser = ap.ArgumentParser(
		description="Reads dicom file and pulls metadata",
		epilog="Version 0.1; Created by Rohan Shad, MD"
	)

	parser.add_argument('-r', '--root_dir', metavar='', required=False, help='Full path to root directory', default='/Users/rohanshad/PHI Safe/test_mri_downloads/archive')
	parser.add_argument('-l', '--csv_list', metavar='', required=False, help='Process only files listed in csv_list.csv', default=None)
	parser.add_argument('-o', '--output_dir', metavar='', default=os.getcwd(), required=False, help='Where all output files will be stored')
	parser.add_argument('-c', '--cpus', metavar='', type=int, default='4',help='number of cores to use in multiprocessing')
	parser.add_argument('-s', '--summarize', metavar='', type=bool, default=False, help='Summarize with frequency counts table in the end')
	parser.add_argument('-i', '--institution', metavar='', required=True, help='institution name')


	args = vars(parser.parse_args())
	print(args)

	root_dir = args['root_dir']
	csv_list = args['csv_list']
	cpus = args['cpus']
	summarize = args['summarize']
	output_dir = args['output_dir']
	institution_prefix = args['institution']
	os.makedirs(output_dir, exist_ok=True)


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
	dicom_metadata_scanner = Dicom_Metadata_Scanner(root_dir, output_dir, institution_prefix)
	for f in filenames:
		# Only loops through tgz files
		if f[-3:] == 'tgz':
			if cpus > 1:
				async_results.append(p.apply_async(dicom_metadata_scanner.process_dicoms, [f]))
				#print(async_result.get())
				
			else:
				async_results.append(dicom_metadata_scanner.process_dicoms(f))

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
	metadata_df = pd.DataFrame(final_list, columns=['series_description', 'frame_loc', 'scanner', 'field_strength','mrn','accession'])
	print(metadata_df)
	metadata_df.to_csv(f'{institution_prefix}_metadata.csv', index=False)

	if summarize:
		freq = metadata_df['series_description'].value_counts().reset_index()
		freq.columns = ['series_description', 'frequency']
		print(freq)
		freq.to_csv(f'{institution_prefix}_metadata_freq.csv', index=False)
	

	print()
	print('Saved metadata from', len(async_results), 'files (Accession numbers)')
	print('Total number of views:', len(final_list))
	print('Elapsed time:', round((time.time() - start_time), 2))
	print('------------------------------------')

