'''
dicom_metadata.py — Extract and catalog DICOM metadata from tar.gz archives.

Scans a directory of .tgz DICOM archives and extracts key metadata fields from
each series: SeriesDescription, SliceLocation, Manufacturer, MagneticFieldStrength,
PatientID (MRN), and AccessionNumber. Output is written to a CSV file for use in
building series name mappings and dataset composition analysis.

Supports institution-specific identifier overrides for medstar, segmed, and ukbiobank,
where standard DICOM tag fields may be blank or formatted differently.

Primary use case: generating the raw series description inventory needed to build
or extend a series_descriptions mapping for your institution.

Usage:
    python dicom_metadata.py -r /path/to/dicoms -i stanford -o /path/to/output
    python dicom_metadata.py -r /path/to/dicoms -i stanford -s          # with frequency summary
    python dicom_metadata.py -r /path/to/dicoms -i stanford \\
        -f mrn:PatientID accession:AccessionNumber scanner:Manufacturer  # custom fields
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
from local_config import get_cfg

_cfg     = get_cfg()
ATTN_DIR = _cfg.attn_dir
TMP_DIR  = _cfg.tmp_dir


class Dicom_Metadata_Scanner:
	'''
	Extract metadata from DICOM series stored in tar.gz archives.

	For each archive, extracts to TMP_DIR, iterates series subfolders,
	reads metadata from a sample DICOM per subfolder, and returns structured
	records for downstream CSV export. Cleans up extracted files after processing.

	The set of extracted fields is configurable via the `fields` dict, which maps
	output column names to DICOM tag attribute names. This determines both what is
	extracted and the column names in the output DataFrame.

	Args:
		root_dir:            Path to directory containing .tgz archives.
		output_dir:          Path where output CSVs will be written.
		institution_prefix:  Institution identifier used for filename-based
		                     MRN/accession overrides (e.g. 'medstar', 'ukbiobank').
		fields:              Dict mapping output column name → DICOM tag attribute name.
		                     e.g. {'mrn': 'PatientID', 'scanner': 'Manufacturer'}
	'''
	def __init__(self, root_dir, output_dir, institution_prefix, fields):
		self.root_dir = root_dir
		self.output_dir = output_dir
		self.institution_prefix = institution_prefix
		self.fields = fields

	def dcm_reader(self, dcm_subfolder):
		'''
		Read metadata from the first DICOM file in a series subfolder.

		Iterates self.fields to extract each requested DICOM tag. All values are
		cast to str to handle pydicom types (PersonName, DS, IS, etc.) that do not
		serialize cleanly as raw objects. Tags absent on a given DICOM are recorded
		as None. SeriesDescription values are sanitized (spaces and slashes replaced
		with underscores). Applies institution-specific overrides for medstar, segmed,
		and ukbiobank on the 'mrn' and 'accession' keys, which are always present
		via FIXED_FIELDS.

		Args:
			dcm_subfolder: Path to a single series directory containing .dcm files.

		Returns:
			Dict of {column_name: value} for all fields in self.fields,
			or None if the DICOM is corrupted or unreadable.
		'''
		dicom_list = os.listdir(dcm_subfolder)

		try:
			df = dcm.dcmread(os.path.join(dcm_subfolder, dicom_list[1]))
			df.PhotometricInterpretation = 'MONOCHROME2'

			record = {}
			for col_name, tag in self.fields.items():
				try:
					value = getattr(df, tag)
					# Cast to str to handle pydicom types (PersonName, DS, IS, etc.)
					# that do not serialize cleanly as raw objects
					value = str(value)
					if tag == 'SeriesDescription':
						value = value.replace(' ', '_').replace('/', '_')
					record[col_name] = value
				except AttributeError:
					record[col_name] = None

			# Institution-specific MRN/accession overrides
			if self.institution_prefix in ('medstar', 'segmed', 'ukbiobank'):
				if 'mrn' in record:
					record['mrn'] = self.filename.split('-')[0]
				if 'accession' in record:
					record['accession'] = self.filename.split('-')[1][:-4]

			return record

		except Exception as ex:
			print("DICOM corrupted! Skipping...")
			print(ex)


	def process_dicoms(self, filename):
		'''
		Extract a single .tgz archive and collect metadata from all series subfolders.

		Extracts to TMP_DIR, globs all second-level subdirectories (series folders),
		calls dcm_reader() on each non-empty folder, and returns a flat list of
		metadata tuples. Cleans up the extracted directory after processing.
		Designed to be called via multiprocessing.Pool.apply_async().

		Args:
			filename: Basename of the .tgz archive within root_dir.

		Returns:
			List of dicts, one per series subfolder, with keys matching self.fields.
			Entries for corrupted DICOMs are None and filtered out by the caller.
		'''
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
			if len(glob.glob(os.path.join(glob.escape(dcm_subfolder), '*'))) > 0:
				metadata_minilist.append(self.dcm_reader(dcm_subfolder))
			else:
				print('Skipped empty dir series')

		# Clean up after to save space	
		try:
			rmtree(tar_extract_path, ignore_errors=True)
		except Exception as ex:
			print('Failed to purge TMP_DIR')
		
		print('Completed processing', filename)
		
		return metadata_minilist



if __name__ == '__main__':

	# Always extracted — institution-specific overrides depend on these being present.
	FIXED_FIELDS = {
		'mrn':       'PatientID',
		'accession': 'AccessionNumber',
	}

	# Additional metadata fields: maps output column name → DICOM tag attribute name.
	# Edit this dict to change what else is extracted. Any valid pydicom attribute works.
	# Use -f at the CLI to append fields without editing this file.
	FIELDS = {
		'series_description': 'SeriesDescription',
		'frame_loc':          'SliceLocation',
		'scanner':            'Manufacturer',
		'field_strength':     'MagneticFieldStrength',
	}

	parser = ap.ArgumentParser(
		description="Reads dicom file and pulls metadata",
		epilog="Version 0.1; Created by Rohan Shad, MD"
	)

	parser.add_argument('-r', '--root_dir', metavar='', required=False, help='Full path to root directory', default='/Users/rohanshad/PHI Safe/test_mri_downloads/archive')
	parser.add_argument('-l', '--csv_list', metavar='', required=False, help='Process only files listed in csv_list.csv', default=None)
	parser.add_argument('-o', '--output_dir', metavar='', default=os.getcwd(), required=False, help='Where all output files will be stored')
	parser.add_argument('-c', '--cpus', metavar='', type=int, default='4', help='number of cores to use in multiprocessing')
	parser.add_argument('-s', '--summarize', action='store_true', default=False, help='Summarize with frequency counts table in the end')
	parser.add_argument('-i', '--institution', metavar='', required=True, help='institution name')
	parser.add_argument('-f', '--fields', metavar='', nargs='+', default=None,
		help='Override metadata fields as column_name:DicomTag pairs, e.g. mrn:PatientID scanner:Manufacturer')

	args = vars(parser.parse_args())
	print(args)

	root_dir         = args['root_dir']
	csv_list         = args['csv_list']
	cpus             = args['cpus']
	summarize        = args['summarize']
	output_dir       = args['output_dir']
	institution_prefix = args['institution']
	os.makedirs(output_dir, exist_ok=True)

	# -f appends to FIELDS (cannot override FIXED_FIELDS)
	if args['fields'] is not None:
		FIELDS = {}
		for pair in args['fields']:
			col_name, tag = pair.split(':')
			col_name, tag = col_name.strip(), tag.strip()
			if col_name not in FIXED_FIELDS:
				FIELDS[col_name] = tag

	# Merge: FIXED_FIELDS always first, then configurable FIELDS
	all_fields = {**FIXED_FIELDS, **FIELDS}

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
	dicom_metadata_scanner = Dicom_Metadata_Scanner(root_dir, output_dir, institution_prefix, all_fields)
	for f in filenames:
		if f[-3:] == 'tgz':
			if cpus > 1:
				async_results.append(p.apply_async(dicom_metadata_scanner.process_dicoms, [f]))
			else:
				async_results.append(dicom_metadata_scanner.process_dicoms(f))
		else:
			continue

	p.close()
	p.join()

	print('------------------------------------')
	print('Collecting results...')
	final_list = []
	for i in async_results:
		sublist = i.get()
		for item in sublist:
			if item is not None:
				final_list.append(item)

	# Column names are derived directly from all_fields keys
	metadata_df = pd.DataFrame(final_list)
	# Attempt to collapse to one row per MRN if fields are constant within MRN
	mrn_df = metadata_df.drop_duplicates(subset=['mrn'])
	if len(mrn_df) < len(metadata_df):
		print(f'Grouped to {len(mrn_df)} unique MRNs (from {len(metadata_df)} series rows)')
		print(mrn_df)
		mrn_df.to_csv(os.path.join(output_dir,f'{institution_prefix}_metadata.csv'), index=False)
	else:
		print(metadata_df)
		metadata_df.to_csv(os.path.join(output_dir,f'{institution_prefix}_metadata.csv'), index=False)

	if summarize:
		if 'series_description' in metadata_df.columns:
			freq = metadata_df['series_description'].value_counts().reset_index()
			freq.columns = ['series_description', 'frequency']
			print(freq)
			freq.to_csv(f'{institution_prefix}_metadata_freq.csv', index=False)

		summarize_cols = ['accession', 'field_strength', 'scanner']
		available = [c for c in summarize_cols if c in metadata_df.columns]
		if available:
			meta_df = metadata_df.drop_duplicates('accession') if 'accession' in metadata_df.columns else metadata_df
			if 'mrn' in meta_df.columns:
				meta_df = meta_df.copy()
				meta_df['parent_folder'] = meta_df['mrn']
				available = ['parent_folder'] + available
			meta_df['institution'] = institution_prefix
			meta_df = meta_df[available + ['institution']]
			meta_df.to_csv(f'{institution_prefix}_meta_df_summarized.csv', index=False)

			for col in ('scanner', 'field_strength'):
				if col in meta_df.columns:
					print('------------------------------------')
					print(meta_df[col].value_counts(dropna=False))

	print()
	print('Saved metadata from', len(async_results), 'files (Accession numbers)')
	print('Total number of views:', len(final_list))
	print('Elapsed time:', round((time.time() - start_time), 2))
	print('------------------------------------')

