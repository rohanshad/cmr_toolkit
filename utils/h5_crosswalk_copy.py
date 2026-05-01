'''
h5_crosswalk_copy.py — Copy HDF5 datastore tree with crosswalk-renamed patient folders.

Reads a crosswalk CSV with columns [anon_mrn, anon_accession, crosswalk_mrn, crosswalk_accession]
and copies matched HDF5 files into a new directory tree with both the patient folder and
accession filename replaced by their crosswalk counterparts. Only rows where both
crosswalk_mrn and crosswalk_accession are non-null are copied.

Source structure:  source_dir/{institution_prefix}_{anon_mrn}/{anon_accession}.h5
Output structure:  output_dir/{crosswalk_mrn}/{crosswalk_accession}.h5

Institution prefix is resolved automatically via glob — no institution arg required.
Each copied file is checksum-validated against its source; failed copies are deleted
and reported. Supports multiprocessing for large crosswalk tables.

Usage:
    python h5_crosswalk_copy.py -s /path/to/hdf5_store -o /path/to/output -f crosswalk.csv
    python h5_crosswalk_copy.py -s /path/to/hdf5_store -o /path/to/output -f crosswalk.csv -c 1
'''

import os
import glob
import shutil
import multiprocessing
import time
import argparse as ap
import pandas as pd
from generate_checksums import compute_checksum


def copy_and_validate(row, source_dir, output_dir):
	'''
	Copy a single HDF5 file to the crosswalk-renamed destination and validate integrity.

	Resolves the source path by globbing for any institution prefix matching the
	anon_mrn. Copies with shutil.copy2 to preserve file metadata, then compares
	source and destination dataset-level SHA256 checksums. If the checksums differ,
	the corrupted destination file is deleted and the row is flagged as failed.

	Args:
		row:        Dict with keys anon_mrn, anon_accession, crosswalk_mrn, crosswalk_accession.
		source_dir: Root directory containing {institution_prefix}_{anon_mrn}/ subdirs.
		output_dir: Destination root; output written to output_dir/{crosswalk_mrn}/.

	Returns:
		Tuple of (status, label) where status is one of:
		'copied', 'not_found', 'ambiguous', 'checksum_fail', 'error'.
	'''
	anon_mrn             = str(row['anon_mrn'])
	anon_accession       = str(row['anon_accession'])
	crosswalk_mrn        = str(row['crosswalk_mrn'])
	crosswalk_accession  = str(row['crosswalk_accession'])

	src_label = f'{anon_mrn}/{anon_accession}.h5'
	dst_label = f'{crosswalk_mrn}/{crosswalk_accession}.h5'

	try:
		# Glob resolves any institution prefix without needing to know it explicitly.
		# Escape source_dir, anon_mrn, and anon_accession to handle special characters
		# (spaces, brackets, etc.) — UK Biobank MRNs are known to contain spaces.
		matches = glob.glob(os.path.join(
			glob.escape(source_dir),
			f'*_{glob.escape(anon_mrn)}',
			f'{glob.escape(anon_accession)}.h5',
		))

		if not matches:
			print(f'NOT FOUND: {src_label}')
			return ('not_found', src_label)

		if len(matches) > 1:
			# Multiple institution folders matched the same anon_mrn — flag as a
			# data-integrity issue rather than silently picking the first one.
			print(f'AMBIGUOUS: {src_label} matched {len(matches)} folders: {matches}')
			return ('ambiguous', src_label)

		src_path = matches[0]
		dst_dir  = os.path.join(output_dir, crosswalk_mrn)
		dst_path = os.path.join(dst_dir, f'{crosswalk_accession}.h5')

		os.makedirs(dst_dir, exist_ok=True)
		shutil.copy2(src_path, dst_path)

		# Validate dataset-level checksum — robust to HDF5 header differences
		_, src_checksum = compute_checksum(src_path)
		_, dst_checksum = compute_checksum(dst_path)

		if src_checksum != dst_checksum:
			print(f'CHECKSUM FAIL: {src_label} — deleting corrupted copy')
			os.remove(dst_path)
			return ('checksum_fail', src_label)

		print(f'Copied: {src_label} → {dst_label}')
		return ('copied', f'{src_label} → {dst_label}')

	except Exception as ex:
		print(f'ERROR: {src_label} — {ex}')
		return ('error', src_label)


if __name__ == '__main__':

	parser = ap.ArgumentParser(
		description='Copy HDF5 datastore tree with crosswalk-renamed patient folders.',
		epilog='Version 1.0; Created by Rohan Shad, MD'
	)

	parser.add_argument('-s', '--source_dir', metavar='', required=True,
		help='Root directory containing institution_mrn/ HDF5 subdirs')
	parser.add_argument('-o', '--output_dir', metavar='', required=True,
		help='Destination root for crosswalk-renamed output tree')
	parser.add_argument('-f', '--crosswalk_csv', metavar='', required=True,
		help='CSV with columns: anon_mrn, anon_accession, crosswalk_mrn, crosswalk_accession')
	parser.add_argument('-c', '--cpus', metavar='', type=int, default=4,
		help='Number of cores to use (default: 4; use 1 for serial/debug mode)')

	args = vars(parser.parse_args())
	print(args)

	source_dir    = args['source_dir']
	output_dir    = args['output_dir']
	crosswalk_csv = args['crosswalk_csv']
	cpus          = args['cpus']

	os.makedirs(output_dir, exist_ok=True)

	# Force dtype=str so all-numeric MRNs/accessions are not coerced to float
	# (which would yield filenames like "12345.0.h5" when NaNs trigger float upcast)
	df = pd.read_csv(crosswalk_csv, dtype=str)

	# Validate required columns up front so missing columns yield a clear error
	# instead of an opaque KeyError deep inside dropna.
	required_cols = {'anon_mrn', 'anon_accession', 'crosswalk_mrn', 'crosswalk_accession'}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f'CSV missing required columns: {sorted(missing)}')

	total_rows = len(df)
	df = df.dropna(subset=['crosswalk_mrn', 'crosswalk_accession']).reset_index(drop=True)
	skipped_count = total_rows - len(df)

	# Detect duplicate destinations — two rows targeting the same crosswalk path
	# would silently overwrite each other in shutil.copy2.
	dupes = df[df.duplicated(subset=['crosswalk_mrn', 'crosswalk_accession'], keep=False)]
	if len(dupes):
		print('------------------------------------')
		print(f'WARN: {len(dupes)} rows have colliding (crosswalk_mrn, crosswalk_accession) pairs:')
		print(dupes[['anon_mrn', 'anon_accession', 'crosswalk_mrn', 'crosswalk_accession']])
		print('Aborting — resolve duplicates in the CSV and rerun.')
		raise SystemExit(1)

	print('------------------------------------')
	print(f'Total rows in crosswalk CSV: {total_rows}')
	print(f'Skipped (no crosswalk IDs):  {skipped_count}')
	print(f'To process:                  {len(df)}')
	print(f'CPUs:                        {cpus}')
	print('------------------------------------')

	rows = df.to_dict('records')
	start_time = time.time()

	if cpus > 1:
		p = multiprocessing.Pool(processes=cpus)
		async_results = [
			p.apply_async(copy_and_validate, [row, source_dir, output_dir])
			for row in rows
		]
		p.close()
		p.join()
		results = [r.get() for r in async_results]
	else:
		results = [copy_and_validate(row, source_dir, output_dir) for row in rows]

	# Tally results
	copied         = [f for status, f in results if status == 'copied']
	not_found      = [f for status, f in results if status == 'not_found']
	ambiguous      = [f for status, f in results if status == 'ambiguous']
	checksum_fails = [f for status, f in results if status == 'checksum_fail']
	errors         = [f for status, f in results if status == 'error']

	elapsed  = round(time.time() - start_time, 2)
	run_date = time.strftime('%Y-%m-%d')
	log_path = os.path.join(output_dir, f'crosswalk_copy_{run_date}.log')

	summary_lines = [
		f'# crosswalk copy run — {time.strftime("%Y-%m-%d %H:%M:%S")}',
		f'# source:  {source_dir}',
		f'# output:  {output_dir}',
		f'# csv:     {crosswalk_csv}',
		f'elapsed={elapsed}s  copied={len(copied)}  skipped={skipped_count}  '
		f'not_found={len(not_found)}  ambiguous={len(ambiguous)}  '
		f'checksum_fails={len(checksum_fails)}  errors={len(errors)}',
	]

	detail_lines = []
	for f in copied:
		detail_lines.append(f'COPIED\t{f}')
	for f in not_found:
		detail_lines.append(f'NOT_FOUND\t{f}')
	for f in ambiguous:
		detail_lines.append(f'AMBIGUOUS\t{f}')
	for f in checksum_fails:
		detail_lines.append(f'CHECKSUM_FAIL\t{f}')
	for f in errors:
		detail_lines.append(f'ERROR\t{f}')

	with open(log_path, 'a') as log:
		log.write('\n'.join(summary_lines) + '\n')
		log.write('\n'.join(detail_lines) + '\n')

	print('------------------------------------')
	print(f'Elapsed time:    {elapsed}s')
	print(f'Copied:          {len(copied)}')
	print(f'Skipped (no ID): {skipped_count}')
	print(f'Not found:       {len(not_found)}')
	print(f'Ambiguous:       {len(ambiguous)}')
	print(f'Checksum fails:  {len(checksum_fails)}')
	print(f'Errors:          {len(errors)}')
	print(f'Log written to:  {log_path}')

	if not_found or ambiguous or checksum_fails or errors:
		print('------------------------------------')
	if not_found:
		print('NOT FOUND:')
		for f in not_found:
			print(f'  {f}')
	if ambiguous:
		print('AMBIGUOUS (multiple source matches):')
		for f in ambiguous:
			print(f'  {f}')
	if checksum_fails:
		print('CHECKSUM FAILURES:')
		for f in checksum_fails:
			print(f'  {f}')
	if errors:
		print('ERRORS:')
		for f in errors:
			print(f'  {f}')

	print('------------------------------------')
