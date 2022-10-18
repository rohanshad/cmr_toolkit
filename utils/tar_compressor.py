'''
Dumb tar compressor utilitiy
This is a workaround because I don't want to make extensive edits to preprocess_mri.py to deal with non-tar compressed dicoms
'''

import tarfile
import os
import multiprocessing
import argparse as ap
import pydicom as dcm
import time
import glob
import pandas as pd
import shutil
import bcolors

from pyaml_env import BaseConfig, parse_config
from pathlib import Path

# Read local_config.yaml for local variables
cfg = BaseConfig(parse_config(Path(__file__).parent.resolve().parent.joinpath('local_config.yaml')))
TMP_DIR = cfg.tmp_dir

def csv_tarcompress(root_dir, filename, output_dir, csv_reference):
	'''
	First reads dcm files and renames the filename to this
	Then compresses folders into tarfiles
	'''

	ref_data = pd.read_csv(csv_reference)
	dicom_list = glob.glob(os.path.join(root_dir, filename,'*','*'))

	try:
		df = dcm.dcmread(dicom_list[1])

		# Check if any dicoms have non greyscale
		df.PhotometricInterpretation = 'MONOCHROME2'

		# Save Study Instance ID
		study_id = df.StudyInstanceUID
		if study_id in ref_data['study_instance_uid'].to_string():
			mrn =  ref_data.loc[ref_data['study_instance_uid'] == study_id, 'anon_mrn'].values[0]
			accession = ref_data.loc[ref_data['study_instance_uid'] == study_id, 'accession'].values[0]
			print(f'Processing: {mrn}-{accession}')
		else:
			pass

		# Dump entire thing as a tarfile with anonymized mrn as basename
		folder_name = os.path.join(root_dir, filename)
		tar = tarfile.open(os.path.join(output_dir, mrn+'-'+accession+'.tgz'), "w:gz")
		tar.add(folder_name, arcname=filename)
		tar.close()

	except Exception as e:
	 	print("DICOM corrupted! Skipping...")
	 	print(e)

def dcm_tarcompress(root_dir, filename, output_dir):
	'''
	First reads dcm files and renames the filename to this
	Then compresses folders into tarfiles
	'''

	dicom_list = glob.glob(os.path.join(root_dir,filename,'*','*'))

	try:
		#print(dicom_list[1])
		df = dcm.dcmread(dicom_list[1])

		# Check if any dicoms have non greyscale
		df.PhotometricInterpretation = 'MONOCHROME2'

		# Save series name + frame location
		accession = df.AccessionNumber
		mrn = df.PatientID
		print('Processing:', mrn+'-'+accession)

		# Dump entire thing as a tarfile with anonymized mrn as basename
		folder_name = os.path.join(root_dir, filename)
		tar = tarfile.open(os.path.join(output_dir, mrn+'-'+accession+'.tgz'), "w:gz")
		tar.add(folder_name, arcname=filename)
		tar.close()

	except:
	 	print("DICOM corrupted! Skipping...")


def simple_tarcompress(root_dir, filename, output_dir):
	'''
	Compresses folders directly into tarfile without renaming anything
	'''
	folder_name = os.path.join(root_dir, filename)
	print('Processing:', filename)
	tar = tarfile.open(os.path.join(output_dir, filename+'.tgz'), "w:gz")
	tar.add(folder_name, arcname=filename)
	tar.close()


def nl_tarcompress(root_dir, filename, output_dir):
	'''
	Deals with dicom folders without any subfolder strcuture for seriesdescription
	'''

	dicom_list = glob.glob(os.path.join(root_dir, filename, '*', '*'))

	counter = 0
	for i in dicom_list:
		try:
			df = dcm.dcmread(i, force=True)
			series_description = df.SeriesDescription
			series_description = series_description.replace(" ","_")
			series_description = series_description.replace("/","_")
			series_description = series_description.replace("\\", "_") # windows compatibility: remove directory separator
			series_description = series_description.replace(":","_") # windows compatibility: remove illegal character for filename
			# if file has no series description, set placeholder (otherwise folder structure is incorrect)
			if series_description == "":
				series_description = 'no_series_description'

			dicom_basename = os.path.split(i)[1]
			accession_number = os.path.split(os.path.split(i)[0])[1]

			os.makedirs(os.path.join(TMP_DIR, filename, accession_number, series_description), exist_ok=True)
			shutil.copy(i, os.path.join(TMP_DIR, filename, accession_number, series_description, dicom_basename))
			counter = counter + 1
		except Exception as ex:
			print(f'Error code: {ex}')
			print(f'DICOM corrupted! Skipping...')

	# Final compression
	simple_tarcompress(TMP_DIR, filename, output_dir)
	shutil.rmtree(os.path.join(TMP_DIR, filename), ignore_errors=True)
	print(f'{bcolors.OK}Successfully exported {counter} dicom files to tar{bcolors.END}')



if __name__ == '__main__':


	parser = ap.ArgumentParser(
		description="Tar compress uncompressed data",
		epilog="Version 0.1; Created by Rohan Shad, MD"
	)
	parser.add_argument('-r', '--root_dir', metavar='', required=False, help='Full path to root directory', default='/scratch/groups/willhies/ukbb_test/bulk_data/raw_data_dump')
	parser.add_argument('-l', '--csv_ref', metavar='', required=False, help='Anonymize via csv reference sheet', default=None)
	parser.add_argument('-o', '--output_dir', metavar='', required=False, help='Where all output files will be stored', default='/scratch/groups/willhies/ukbb_test/bulk_data/tar_outputs')
	parser.add_argument('-c', '--cpus', metavar='', type=int, default='4',help='number of cores to use in multiprocessing')
	parser.add_argument('-d', '--debug', action='store_true', default=False)
	parser.add_argument('-m', '--mode', metavar='', type=str, default='simple')

	args = vars(parser.parse_args())
	print(args)


	root_dir = args['root_dir']
	csv_reference = args['csv_ref']
	cpus = args['cpus']
	debug = args['debug']
	mode = args['mode']
	output_dir = args['output_dir']
	os.makedirs(output_dir, exist_ok=True)

	# Start worker pool
	p = multiprocessing.Pool(processes=cpus)

	start_time = time.time()
	filenames = os.listdir(root_dir)
	filenames = [i for i in filenames if i[0] != "."]

	if mode == 'simple':
		'''
		Use this to just tar compress preserving dicom folder name
		'''
		for f in filenames:
			if cpus > 1:
				p.apply_async(simple_tarcompress, [root_dir, f, output_dir])
			else:
				simple_tarcompress(root_dir, f, output_dir)

	elif mode == 'dicom':
		'''
		Use if reading MRN and Accession numbers from dicom files and using them directly
		(Assumes your files are anonymized, please for the love of god don't do this if they aren't)
		'''
		for f in filenames:
			if cpus > 1:
				p.apply_async(dcm_tarcompress, [root_dir, f, output_dir])
			else:
				dcm_tarcompress(root_dir, f, output_dir)

	elif mode == 'anonymize':
		'''
		Use to read MRN and Accession numbers from dicom files and anonymize them based on a provided crosswalk to provide
		anon-MRN and anon-accession numbers. Crosswalk is provided via --csv_ref flag
		'''
		for f in filenames:
			if cpus > 1:
				p.apply_async(csv_tarcompress, [root_dir, f, output_dir, csv_reference])
			else:
				csv_tarcompress(root_dir, f, output_dir, csv_reference)

	elif mode == 'nl_tarcompress':

		for f in filenames:
			if cpus > 1:
				p.apply_async(nl_tarcompress, [root_dir, f, output_dir])
			else:
				nl_tarcompress(root_dir, f, output_dir)

	p.close()
	p.join()

	print('Elapsed time:', round((time.time() - start_time), 2))
