'''
dicom_metadata.py — Extract and catalog DICOM metadata from tar.gz archives.

Extraction granularity is set with --decompose:
    study   one record per archive, sampled from the first series (default)
    series  one record per series folder
    frame   every DICOM, as a nested study -> series -> frames JSON tree with
            h5-aligned frame ordering plus slice_frames / total_images per series

study and series write a flat CSV; frame writes nested JSON. Institution-specific
mrn/accession overrides are applied for medstar, segmed, and ukbiobank.

Primary use case: generating the raw series description inventory needed to build
or extend a series_descriptions mapping for your institution.

Usage:
    python dicom_metadata.py -r /path/to/dicoms -i stanford -o /path/to/output
    python dicom_metadata.py -r /path/to/dicoms -i stanford -d frame       # nested JSON
    python dicom_metadata.py -r /path/to/dicoms -i stanford -s             # frequency summary
    python dicom_metadata.py -r /path/to/dicoms -i stanford -f scanner:Manufacturer
'''

import torch
import matplotlib
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import argparse as ap
import glob
import json
import pydicom as dcm
import time
import multiprocessing
import tarfile
from shutil import rmtree
import pandas as pd
from collections import Counter
from natsort import natsorted, natsort_keygen
from local_config import get_cfg

_cfg     = get_cfg()
ATTN_DIR = _cfg.attn_dir
TMP_DIR  = _cfg.tmp_dir


# Always extracted — institution-specific overrides depend on these being present.
FIXED_FIELDS = {
	'mrn':       'PatientID',
	'accession': 'AccessionNumber',
}

# Field tiers used by --decompose frame. Each tier maps output key → DICOM tag
# attribute name, same convention as FIXED_FIELDS / FIELDS. Study and series tiers
# are read once off a series' first file; only FRAME_FIELDS is read per DICOM.
STUDY_FIELDS = {
	'scanner':        'Manufacturer',
	'field_strength': 'MagneticFieldStrength',
}

SERIES_FIELDS = {
	'series_description': 'SeriesDescription',
	'pixel_spacing':      'PixelSpacing',
	'slice_thickness':    'SliceThickness',
}

FRAME_FIELDS = {
	'instance_number': 'InstanceNumber',
	'slice_location':  'SliceLocation',
	'trigger_time':    'TriggerTime',
}


def list_series_files(dcm_subfolder):
	'''
	Naturally sorted, dotfile-free filename listing of a series folder.
	natsorted so IM-2 precedes IM-10; [] if the path is not a directory.
	'''
	if not os.path.isdir(dcm_subfolder):
		return []
	return natsorted(f for f in os.listdir(dcm_subfolder) if not f.startswith('.'))


class Dicom_Metadata_Scanner:
	'''
	Extract metadata from DICOM series stored in tar.gz archives.

	Extracts each archive to TMP_DIR, reads metadata at the granularity set by
	`decompose`, and cleans up afterwards. Study/series levels use the configurable
	`fields` dict; frame level uses the module-level STUDY/SERIES/FRAME_FIELDS
	tiers, snapshotted onto the instance so -f overrides survive spawned workers.

	Args:
		root_dir:            Path to directory containing .tgz archives.
		output_dir:          Path where output files will be written.
		institution_prefix:  Institution id; drives filename-based mrn/accession overrides.
		fields:              Dict of {column_name: DicomTag}, e.g. {'scanner': 'Manufacturer'}.
		decompose:           One of 'study', 'series', 'frame'.
	'''
	def __init__(self, root_dir, output_dir, institution_prefix, fields, decompose='study'):
		self.root_dir = root_dir
		self.output_dir = output_dir
		self.institution_prefix = institution_prefix
		self.fields = fields
		self.decompose = decompose
		# Workers are spawned, not forked, so module globals reset to their defaults
		# in the child. Snapshot the tiers here so -f overrides actually travel.
		self.fixed_fields = FIXED_FIELDS
		self.study_fields = STUDY_FIELDS
		self.series_fields = SERIES_FIELDS
		self.frame_fields = FRAME_FIELDS

	def dcm_reader(self, dcm_path, fields=None):
		'''
		Read metadata (header only) from a single DICOM file.

		Values are str-cast (pydicom types don't serialize cleanly), missing tags
		record as None, and SeriesDescription is sanitized. medstar/segmed/ukbiobank
		mrn/accession are overridden from the tar filename.

		Args:
			dcm_path: Path to a single .dcm file.
			fields:   {column_name: DicomTag} to extract; defaults to self.fields.

		Returns:
			Dict of extracted values, or None if the DICOM is unreadable.
		'''
		fields = self.fields if fields is None else fields

		try:
			df = dcm.dcmread(dcm_path, stop_before_pixels=True)

			record = {}
			for col_name, tag in fields.items():
				try:
					value = getattr(df, tag)
					# Cast to str to handle pydicom types (PersonName, DS, IS, etc.)
					# that do not serialize cleanly. Present-but-empty tags return None.
					if value is not None:
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


	def series_eligible(self, dcm_subfolder, dicom_list):
		'''
		Mirror the folder-level skips in preprocess_mri.view_disambugator: oversized
		folders, unreadable first DICOM / missing SeriesDescription, InlineVF overlays.
		'''
		if len(dicom_list) >= 2500:
			print(f'Insane number of frames detected: {len(dicom_list)}; skipping...')
			return False
		try:
			series = dcm.dcmread(os.path.join(dcm_subfolder, dicom_list[0]), stop_before_pixels=True).SeriesDescription
		except Exception as e:
			print(f'Failed to parse DICOM in {dcm_subfolder}: {e}')
			return False
		if series is None or 'InlineVF' in series:
			print('Skipping series (blank description or InlineVF overlay)...')
			return False
		return True


	def process_dicoms(self, filename):
		'''
		Extract a single .tgz archive and collect metadata at the requested level.

		Extracts to TMP_DIR, iterates series subfolders, cleans up afterwards.
		Designed to be called via multiprocessing.Pool.apply_async().

		'study' reads one DICOM from the first non-empty series; 'series' reads the
		first DICOM of every series; 'frame' reads every DICOM and returns one nested
		dict per archive. Series and frames that preprocess_mri would skip (InlineVF,
		oversized folders, frames missing required tags) are omitted at every level.
		Frame mode orders each 'frames' list by (SliceLocation, InstanceNumber) to
		match the h5 arrays from preprocess_mri.collate_arrays, stamps each frame
		with 'frame_num' (its index in the h5 array), and carries 'slice_frames' /
		'total_images' per series, mirroring the h5 attrs. Alignment is exact for
		single-folder series; multi-folder stacks (ukbiobank SAX) are merged and
		trimmed in the h5 and will not match. Source filenames are omitted (they
		can embed SOPInstanceUIDs).

		Args:
			filename: Basename of the .tgz archive within root_dir.

		Returns:
			List of dicts: flat records for study/series, a single nested dict for
			frame. Corrupted-DICOM entries are None and filtered by the caller.
		'''
		self.filename = filename
		tar = tarfile.open(os.path.join(self.root_dir, self.filename))
		tar_extract_path = os.path.join(TMP_DIR, self.filename[:-4])
		tar.extractall(tar_extract_path)
		tar.close()

		# List series folders and iterate over them all one by one

		print("Extracted tarfile for", self.filename[:-4], "...")

		# Sorted so 'study' level samples the same series on every filesystem
		dcm_directory = sorted(glob.glob(os.path.join(tar_extract_path, '*', '*')))

		metadata_minilist = []

		if self.decompose == 'frame':
			study_record = None
			series_list = []
			# preprocess_mri.dcm_to_array requires these tags on every frame; probe
			# them alongside the frame tier so its per-frame drops can be mirrored
			probe_fields = {**self.frame_fields, '_series': 'SeriesDescription',
				'_accession': 'AccessionNumber', '_mrn': 'PatientID'}

			for dcm_subfolder in dcm_directory:
				dicom_list = list_series_files(dcm_subfolder)
				if len(dicom_list) == 0:
					print('Skipped empty dir series')
					continue
				if not self.series_eligible(dcm_subfolder, dicom_list):
					continue

				first_dcm = os.path.join(dcm_subfolder, dicom_list[0])

				# Study tier is constant across the archive, so read it once off
				# the first series that opens cleanly
				if study_record is None:
					study_record = self.dcm_reader(first_dcm, {**self.fixed_fields, **self.study_fields})

				series_record = self.dcm_reader(first_dcm, self.series_fields)
				if series_record is None:
					continue

				# Frames missing a required tag never reach the h5; drop them here too.
				# Filenames are not recorded as they may embed SOPInstanceUIDs.
				frames = []
				for d in dicom_list:
					frame_record = self.dcm_reader(os.path.join(dcm_subfolder, d), probe_fields)
					if frame_record is None:
						continue
					probes = [frame_record.pop(k) for k in ('_series', '_accession', '_mrn')]
					if None in probes or frame_record['slice_location'] is None:
						continue
					frames.append(frame_record)

				# A series with no surviving frames never reaches the h5 either
				if len(frames) == 0:
					print('No usable frames, skipping series...')
					continue

				# Mirror preprocess_mri.collate_arrays frame ordering + slice_frames so
				# frames[i] lines up with h5 frame i; frame_num makes that index explicit
				tmp_df = pd.DataFrame({
					'unique_frame_index': [f.get('instance_number') for f in frames],
					'slice_location':     [float(f['slice_location']) for f in frames],
				})
				tmp_df = tmp_df.sort_values(by=['slice_location', 'unique_frame_index'], key=natsort_keygen())
				frames = [frames[i] for i in tmp_df.index.tolist()]
				sl = np.array(tmp_df['slice_location'].tolist())
				slice_frames = np.where(sl[:-1] != sl[1:])[0].tolist()
				frames = [{'frame_num': i, **f} for i, f in enumerate(frames)]

				series_record['slice_frames'] = slice_frames
				series_record['total_images'] = len(dicom_list)
				series_record['frames'] = frames
				series_list.append(series_record)

			if study_record is not None:
				metadata_minilist.append({**study_record, 'series': series_list})

		else:
			for dcm_subfolder in dcm_directory:
				dicom_list = list_series_files(dcm_subfolder)
				if len(dicom_list) == 0:
					print('Skipped empty dir series')
					continue
				if not self.series_eligible(dcm_subfolder, dicom_list):
					continue

				metadata_minilist.append(self.dcm_reader(os.path.join(dcm_subfolder, dicom_list[0])))
				# Study level fields are constant across the archive, one series is enough
				if self.decompose == 'study':
					break

		# Clean up after to save space
		try:
			rmtree(tar_extract_path, ignore_errors=True)
		except Exception as ex:
			print('Failed to purge TMP_DIR')

		print('Completed processing', filename)

		return metadata_minilist



if __name__ == '__main__':

	# Additional metadata fields for study and series levels: maps output column name
	# → DICOM tag attribute name. Edit this dict to change what else is extracted. Any
	# valid pydicom attribute works. Use -f at the CLI to override without editing this
	# file. Frame level uses the STUDY_FIELDS / SERIES_FIELDS / FRAME_FIELDS tiers instead.
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
	parser.add_argument('-d', '--decompose', metavar='', choices=['study', 'series', 'frame'], default='study',
		help='Extraction granularity: one record per study, per series, or per frame (frame writes nested JSON)')
	parser.add_argument('-f', '--fields', metavar='', nargs='+', default=None,
		help='Metadata fields as column_name:DicomTag pairs, e.g. scanner:Manufacturer. Replaces the defaults, except at frame level where pairs are appended')

	args = vars(parser.parse_args())
	print(args)

	root_dir         = args['root_dir']
	csv_list         = args['csv_list']
	cpus             = args['cpus']
	summarize        = args['summarize']
	output_dir       = args['output_dir']
	decompose        = args['decompose']
	institution_prefix = args['institution']
	os.makedirs(output_dir, exist_ok=True)

	# -f sets the configurable field set (cannot override FIXED_FIELDS). At frame
	# level it appends to FRAME_FIELDS — the defaults (and computed frame_num) are
	# protected since the h5 alignment depends on them. Elsewhere it replaces FIELDS.
	if args['fields'] is not None:
		override_fields = {}
		for pair in args['fields']:
			col_name, tag = pair.split(':')
			col_name, tag = col_name.strip(), tag.strip()
			if col_name not in FIXED_FIELDS:
				override_fields[col_name] = tag
		if decompose == 'frame':
			protected = set(FRAME_FIELDS) | {'frame_num'}
			FRAME_FIELDS = {**FRAME_FIELDS, **{k: v for k, v in override_fields.items() if k not in protected}}
		else:
			FIELDS = override_fields

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
	dicom_metadata_scanner = Dicom_Metadata_Scanner(root_dir, output_dir, institution_prefix, all_fields, decompose)
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
		# cpus == 1 runs inline, so those entries are plain lists, not AsyncResults
		sublist = i.get() if hasattr(i, 'get') else i
		for item in sublist:
			if item is not None:
				final_list.append(item)

	if decompose == 'frame':
		json_path = os.path.join(output_dir, f'{institution_prefix}_metadata.json')
		with open(json_path, 'w') as fh:
			json.dump(final_list, fh, indent=2)

		series_count = sum(len(record['series']) for record in final_list)
		frame_count = sum(len(series['frames']) for record in final_list for series in record['series'])
		print(f'{len(final_list)} studies / {series_count} series / {frame_count} frames written to {json_path}')

		# Flatten to one row per series so the summary code below stays as-is
		flat_rows = []
		for record in final_list:
			study_cols = {k: v for k, v in record.items() if k != 'series'}
			for series in record['series']:
				series_cols = {k: v for k, v in series.items() if k != 'frames'}
				flat_rows.append({**study_cols, **series_cols})
		metadata_df = pd.DataFrame(flat_rows)
	else:
		# Column names are derived directly from all_fields keys
		metadata_df = pd.DataFrame(final_list)
		unique_mrns = metadata_df['mrn'].nunique() if 'mrn' in metadata_df.columns else 0
		print(f'{len(metadata_df)} rows across {unique_mrns} unique MRNs')
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
