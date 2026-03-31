'''
tar_compressor.py — DICOM folder compression and anonymization utilities.

Reads directories of raw DICOM studies and writes standardized, compressed .tgz archives
named as anon_mrn-anon_accession.tgz. Supports multiple institution-specific workflows
via mode selection:

    simple          — compress folder as-is with no renaming
    dicom           — extract MRN/accession from DICOM tags and rename
    anonymize       — remap identifiers via CSV crosswalk (mrn, accession → anon_mrn, anon_accession)
    segmed          — rewrite DICOM tags in-place before compression (SegMed format)
    ukbiobank       — unzip UKB zip dumps, reorganize by SeriesDescription, rewrite tags
    dasa            — rewrite tags for DASA institution format
    penn            — handle flat DICOM folders with no series subdirectory structure
    reset_dicom_meta — overwrite AccessionNumber in original files before compressing

All modes support multiprocessing via Pool.apply_async().

Usage:
    python tar_compressor.py -r /path/to/dicoms -o /path/to/output -m anonymize -l crosswalk.csv
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
import zipfile

from local_config import get_cfg, get_global_cfg

_cfg        = get_cfg()
TMP_DIR     = _cfg.tmp_dir
BUCKET_NAME = get_global_cfg().bucket_name

def csv_tarcompress(root_dir, filename, output_dir, csv_reference):
	'''
	Compress a DICOM folder to .tgz, remapping identifiers via a CSV crosswalk.

	Reads AccessionNumber from a sample DICOM, looks it up in the crosswalk CSV,
	and uses the mapped anon_mrn and anon_accession as the output filename. Skips
	silently if the accession is not found in the crosswalk.

	Args:
		root_dir:      Path to directory containing DICOM study folders.
		filename:      Name of the study folder to compress.
		output_dir:    Destination for the output .tgz file.
		csv_reference: Path to crosswalk CSV with columns: mrn, accession, anon_mrn, anon_accession.
	'''

	ref_data = pd.read_csv(csv_reference).dropna().reset_index()
	dicom_list = glob.glob(os.path.join(root_dir, filename,'*','*'))

	try:
		df = dcm.dcmread(dicom_list[1])
		# Check if any dicoms have non greyscale 
		df.PhotometricInterpretation = 'MONOCHROME2'

		# Process only if Accession Number is in crosswalk file
		accession = df.AccessionNumber  
		mrn = df.PatientID

		# Possibility of MRN not being in crosswalk, but accession # will always match

		if accession in ref_data['accession'].to_string():
			mrn =  ref_data.loc[ref_data['accession'].astype(int).astype(str) == accession, 'anon_mrn'].values[0]
			accession = ref_data.loc[ref_data['accession'].astype(int).astype(str) == accession, 'anon_accession'].values[0]
			print(f'{bcolors.OK}Processing: {mrn}-{accession}{bcolors.END}')
		else:
			pass
			print(f'{bcolors.ERR}No matching scan data in crosswalk for acc: {accession}{bcolors.END}')

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
	Compress a DICOM folder to .tgz, named using MRN and AccessionNumber from DICOM tags.

	Reads PatientID and AccessionNumber directly from DICOM metadata to construct the
	output filename as mrn-accession.tgz. No crosswalk or anonymization is applied.

	Args:
		root_dir:   Path to directory containing DICOM study folders.
		filename:   Name of the study folder to compress.
		output_dir: Destination for the output .tgz file.
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

def segmed_tarcompress(root_dir, filename, output_dir, csv_reference):
	'''
	Anonymize and compress SegMed DICOM studies, rewriting tags in-place before archiving.

	Looks up StudyInstanceUID in the crosswalk CSV to retrieve anon_mrn and anon_uid.
	Rewrites PatientID and AccessionNumber on every DICOM file in TMP_DIR before
	compressing to .tgz. Cleans up TMP_DIR after compression.

	Args:
		root_dir:      Path to directory containing DICOM study folders.
		filename:      Name of the study folder to compress.
		output_dir:    Destination for the output .tgz file.
		csv_reference: Path to crosswalk CSV with columns: Study ID, anon_mrn, anon_uid.
	'''

	ref_data = pd.read_csv(csv_reference).dropna().reset_index()
	dicom_list = glob.glob(os.path.join(root_dir,filename,'*','*'))
	os.chdir(root_dir)

	try:
		#print(dicom_list[1])
		df = dcm.dcmread(dicom_list[1])

		# Check if any dicoms have non greyscale 
		df.PhotometricInterpretation = 'MONOCHROME2'

		# Save series name + frame location 
		study_uid = str(df.StudyInstanceUID)
		mrn = df.PatientID 

		# Possibility of MRN not being in crosswalk, but accession # will always match
		if study_uid in ref_data['Study ID'].astype(str).values:
			mrn =  ref_data.loc[ref_data['Study ID'].astype(str) == study_uid, 'anon_mrn'].values[0]
			accession = ref_data.loc[ref_data['Study ID'].astype(str) == study_uid, 'anon_uid'].values[0]
			print(f'{bcolors.BLUE}Processing{bcolors.ENDC}: {mrn}-{accession}')
		else:
			pass
			print(f'{bcolors.ERR}No matching scan data in crosswalk for acc: {study_uid}{bcolors.END}')

		for dcm_file in dicom_list:
			tmp_path = os.path.join(TMP_DIR, os.path.relpath(dcm_file))
			os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
			
			df = dcm.dcmread(dcm_file)
			# Check if any dicoms have non greyscale 
			df.PhotometricInterpretation = 'MONOCHROME2'
			df.PatientID = mrn
			df.AccessionNumber = accession
			df.is_little_endian = True
			df.is_implicit_VR = False

			df.save_as(tmp_path, write_like_original=False)	

		# Dump entire thing as a tarfile with anonymized mrn as basename
		folder_name = os.path.join(TMP_DIR, os.path.basename(filename))
		tar = tarfile.open(os.path.join(output_dir, mrn+'-'+accession+'.tgz'), "w:gz")
		tar.add(folder_name, arcname=filename)
		tar.close()

		shutil.rmtree(folder_name)

	except Exception as e:
		print("DICOM corrupted! Skipping...")
		print(e)

def ukb_unzip_and_organize(root_dir, target_prefix):
	# Hunt for all scans with specific target eid prefix 
	DEBUG = False
	all_scans_for_eid = glob.glob(os.path.join(root_dir, f'{target_prefix}*'))

	for scan in all_scans_for_eid:
		dat = str.split(os.path.basename(scan), '_')
		eid = dat[0]
		datafield = dat[1]
		instance = dat[2]

		dest_dir = os.path.join(TMP_DIR, eid, instance, datafield)
		#print(dest_dir)
		os.makedirs(dest_dir, exist_ok=True)

		try:
			with zipfile.ZipFile(scan, 'r') as zf:
				zf.extractall(dest_dir)
			if DEBUG:
				print(f"Successfully extracted {scan} to {dest_dir}")

		except zipfile.BadZipFile:
			print(f"Error: '{zip_file_path}' is not a valid zip file or is corrupted.")
		except FileNotFoundError:
			print(f"Error: The file '{zip_file_path}' was not found.")
		except Exception as e:
			print(f"An unexpected error occurred: {e}")





def ukb_tarcompress(root_dir, filename, output_dir, csv_reference):
	'''
	Anonymize and compress a UK Biobank EID zip dump into per-accession .tgz archives.

	Calls ukb_unzip_and_organize() to extract and reorganize zip files by instance/datafield,
	then for each collected scan: looks up anon_mrn and anon_accession from the crosswalk,
	rewrites PatientID, PatientName, and AccessionNumber on every DICOM, reorganizes files
	into SeriesDescription subdirectories in TMP_DIR, and compresses to .tgz.

	Args:
		root_dir:      Path to directory containing UKB zip files named {eid}_{datafield}_{instance}_0.zip.
		filename:      EID string used to glob all zip files for this participant.
		output_dir:    Destination for output .tgz files.
		csv_reference: Path to crosswalk CSV with columns: f.eid, instance, anon_mrn, anon_accession.
	'''
	ukb_unzip_and_organize(root_dir, filename)
	accession_folders = glob.glob(os.path.join(TMP_DIR, filename, '*'))

	for collected_scans in accession_folders:
		ref_data = pd.read_csv(csv_reference).dropna().reset_index()
		dicom_list = glob.glob(os.path.join(collected_scans,'*','*.dcm'))
		os.chdir(root_dir)
		
		try:
			#print(dicom_list[1])
			df = dcm.dcmread(dicom_list[1])

			# Check if any dicoms have non greyscale 
			df.PhotometricInterpretation = 'MONOCHROME2'

			# Save series name + frame location 
			patient_id = str(filename)
			scan_instance = str(os.path.basename(collected_scans))
			mrn =  ref_data.loc[ref_data['f.eid'].astype(str) == patient_id, 'anon_mrn'].values[0]
			accession = ref_data.loc[(ref_data['f.eid'].astype(str) == patient_id) & (ref_data['instance'].astype(str) == scan_instance), 'anon_accession'].values[0]
			print(f'{bcolors.BLUE}Processing{bcolors.ENDC}: {mrn}-{accession}')
			

			for dcm_file in dicom_list:
				df = dcm.dcmread(dcm_file)
				series = df.SeriesDescription
				
				tmp_path = os.path.join(TMP_DIR, f'{mrn}_{accession}', series, os.path.basename(dcm_file))
				os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
				
				# Check if any dicoms have non greyscale 
				df.PhotometricInterpretation = 'MONOCHROME2'
				df.PatientID = mrn
				df.PatientName = f"redacted_{mrn}"
				df.AccessionNumber = accession
				df.is_little_endian = True
				df.is_implicit_VR = False

				df.save_as(tmp_path, write_like_original=False)	

			# Dump entire thing as a tarfile with anonymized mrn as basename
			folder_name = os.path.join(TMP_DIR, f'{mrn}_{accession}')
			tar = tarfile.open(os.path.join(output_dir, mrn+'-'+accession+'.tgz'), "w:gz")
			tar.add(folder_name, arcname=f'{filename}_{accession}')
			tar.close()

			shutil.rmtree(folder_name)

		except Exception as e:
			print("DICOM corrupted! Skipping...")
			print(e)


	try:
		shutil.rmtree(os.path.join(TMP_DIR,filename))
	except Exception as e:
		print(f"Failed to clear tmp for file: {filename}")

def dasa_tarcompress(root_dir, filename, output_dir, csv_reference):
	'''
	Anonymize and compress DASA DICOM studies, rewriting tags before archiving.

	DASA uses PatientID as the accession number rather than MRN. Looks up anon_mrn
	and anon_accession from the crosswalk CSV via PatientID match, rewrites DICOM
	tags in TMP_DIR, and compresses to .tgz. Cleans up TMP_DIR after compression.

	Args:
		root_dir:      Path to directory containing DICOM study folders.
		filename:      Name of the study folder to compress.
		output_dir:    Destination for the output .tgz file.
		csv_reference: Path to crosswalk CSV with columns: accession, anon_mrn, anon_accession.
	'''

	ref_data = pd.read_csv(csv_reference).dropna().reset_index()
	dicom_list = glob.glob(os.path.join(root_dir,filename,'*','*'))
	os.chdir(root_dir)

	try:
		#print(dicom_list[1])
		df = dcm.dcmread(dicom_list[1])

		# Check if any dicoms have non greyscale 
		df.PhotometricInterpretation = 'MONOCHROME2'

		# Save series name + frame location 
		# Dasa uses accession numbers == patient_id not mrn
		study_uid = str(df.StudyInstanceUID)
		patient_id = df.PatientID 

		mrn =  ref_data.loc[ref_data['accession'].astype(str) == patient_id, 'anon_mrn'].values[0]
		accession = ref_data.loc[ref_data['accession'].astype(str) == patient_id, 'anon_accession'].values[0]
		print(f'{bcolors.BLUE}Processing{bcolors.ENDC}: {mrn}-{accession}')
		

		for dcm_file in dicom_list:
			tmp_path = os.path.join(TMP_DIR, os.path.relpath(dcm_file))
			os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

			df = dcm.dcmread(dcm_file)
			# Check if any dicoms have non greyscale 
			df.PhotometricInterpretation = 'MONOCHROME2'
			df.PatientID = mrn
			df.AccessionNumber = accession
			df.is_little_endian = True
			df.is_implicit_VR = True

			df.save_as(tmp_path, write_like_original=False)	

		# Dump entire thing as a tarfile with anonymized mrn as basename
		folder_name = os.path.join(TMP_DIR, os.path.basename(filename))
		tar = tarfile.open(os.path.join(output_dir, mrn+'-'+accession+'.tgz'), "w:gz")
		tar.add(folder_name, arcname=filename)
		tar.close()

		shutil.rmtree(folder_name)

	except Exception as e:
		print("DICOM corrupted! Skipping...")
		print(e)


def dcm_rewrite_originals_tarcompress(root_dir, filename, output_dir):
	'''
	Overwrite AccessionNumber on original DICOM files in-place, then compress to .tgz.

	Sets AccessionNumber to the literal string 'scandata' on every DICOM file before
	compressing. Output filename is derived from PatientID and the rewritten AccessionNumber.
	Modifies the source files directly — use with caution on non-copied data.

	Args:
		root_dir:   Path to directory containing DICOM study folders.
		filename:   Name of the study folder to process.
		output_dir: Destination for the output .tgz file.
	'''

	dicom_list = glob.glob(os.path.join(root_dir,filename,'*','*'))

	for dcm_file in dicom_list:
		df = dcm.dcmread(dcm_file)
		# Check if any dicoms have non greyscale 
		df.PhotometricInterpretation = 'MONOCHROME2'
		df.AccessionNumber = 'scandata'	

		df.is_little_endian = True
		df.is_implicit_VR = True

		df.save_as(dcm_file, write_like_original=False)	

	print(f'Reset AccessionNumbers for {filename}')	

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
	Compress a folder to .tgz without any renaming or DICOM tag modification.

	Output filename is the original folder name with .tgz appended. Use for cases
	where identifiers are already correct and no anonymization is needed.

	Args:
		root_dir:   Path to directory containing folders to compress.
		filename:   Name of the folder to compress.
		output_dir: Destination for the output .tgz file.
	'''
	folder_name = os.path.join(root_dir, filename)
	print('Processing:', filename)
	tar = tarfile.open(os.path.join(output_dir, filename+'.tgz'), "w:gz")
	tar.add(folder_name, arcname=filename)
	tar.close()


def nofolder_tarcompress(root_dir, filename, output_dir, csv_reference):
	'''
	Handle flat DICOM folders (no series subdirectories) by reorganizing before compression.

	Some institutions (e.g. UPenn) deliver DICOMs in a flat directory with no
	SeriesDescription subfolders. This function reads SeriesDescription from each
	DICOM, creates the expected subfolder structure in TMP_DIR, copies files
	accordingly, then delegates to csv_tarcompress or dcm_tarcompress for final
	archiving and cleanup.

	Args:
		root_dir:      Path to directory containing flat DICOM folders.
		filename:      Name of the flat study folder to process.
		output_dir:    Destination for the output .tgz file.
		csv_reference: Path to crosswalk CSV, or None to use dcm_tarcompress (no anonymization).
	'''

	dicom_list = glob.glob(os.path.join(root_dir, filename, '*'))
	counter = 0
	for i in dicom_list:	
		try:
			df = dcm.dcmread(i, force=True)
			series_description = df.SeriesDescription  

			dicom_basename = os.path.split(i)[1]
			accession_number = os.path.split(os.path.split(i)[0])[1]
			series_description = series_description.replace(' ','_').replace('/','_')
			os.makedirs(os.path.join(TMP_DIR, filename, series_description), exist_ok=True)
			shutil.copy(i, os.path.join(TMP_DIR, filename, series_description, dicom_basename))
			counter = counter + 1

		except Exception as ex:
			print(f'Error code: {ex}')
			print(f'DICOM corrupted! Skipping...')

	# Final compression
	if csv_reference is not None:
		csv_tarcompress(TMP_DIR, filename, output_dir, csv_reference)
	else:
		dcm_tarcompress(TMP_DIR, filename, output_dir)
	shutil.rmtree(os.path.join(TMP_DIR, filename))
	print(f'Successfully exported {counter} dicom files to tar')


if __name__ == '__main__':
	

	parser = ap.ArgumentParser(
		description="Tar compress uncompressed data",
		epilog="Version 2.0; Created by Rohan Shad, MD"
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
		for f in filenames:
			if cpus > 1:
				p.apply_async(simple_tarcompress, [root_dir, f, output_dir])
			else:
				simple_tarcompress(root_dir, f, output_dir)


	elif mode == 'dicom':
		for f in filenames:
			if cpus > 1:
				p.apply_async(dcm_tarcompress, [root_dir, f, output_dir])
			else:
				dcm_tarcompress(root_dir, f, output_dir)

	elif mode == 'anonymize':
		for f in filenames:
			if cpus > 1:
				p.apply_async(csv_tarcompress, [root_dir, f, output_dir, csv_reference])
			else:
				csv_tarcompress(root_dir, f, output_dir, csv_reference)

	elif mode == 'reset_dicom_meta':
		for f in filenames:
			if cpus > 1:
				p.apply_async(dcm_rewrite_originals_tarcompress, [root_dir, f, output_dir])
			else:
				dcm_rewrite_originals_tarcompress(root_dir, f, output_dir)

	elif mode == 'penn':
		for f in filenames:
			if cpus > 1:
				p.apply_async(nofolder_tarcompress, [root_dir, f, output_dir, csv_reference])
			else:
				nofolder_tarcompress(root_dir, f, output_dir, csv_reference)	

	elif mode == 'segmed':

		for f in filenames:
			if cpus > 1:
				p.apply_async(segmed_tarcompress, [root_dir, f, output_dir, csv_reference])
			else:
				segmed_tarcompress(root_dir, f, output_dir, csv_reference)	

	elif mode == 'dasa':

		for f in filenames:
			if cpus > 1:
				p.apply_async(dasa_tarcompress, [root_dir, f, output_dir, csv_reference])
			else:
				dasa_tarcompress(root_dir, f, output_dir, csv_reference)	

	elif mode == 'ukbiobank':
		df = pd.read_csv(csv_reference)
		filenames = df['f.eid'].unique().astype('str')
		for f in filenames:
			if cpus > 1:
				p.apply_async(ukb_tarcompress, [root_dir, f, output_dir, csv_reference])
			else:
				ukb_tarcompress(root_dir, f, output_dir, csv_reference)	


	p.close()
	p.join()

	print('Elapsed time:', round((time.time() - start_time), 2))			

