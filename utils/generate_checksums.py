'''
Generate sha256 checksums for pre-processed hdf5 files 
'''

import os
import hashlib
import pandas as pd
import h5py
import argparse as ap
import multiprocessing
import time

def compute_checksum(hdf5_file):
	'''
	Compute the SHA256 checksum of datasets within hdf5 file.
	Simply computing hash on the file itself is not enough, headers and minor metadata differences
	make for differing sha256 between identical processing runs
	'''
	sha256_hash = hashlib.sha256()
	with h5py.File(hdf5_file, "r") as f:
		datasets = sorted(f.keys())  # Sort dataset names for consistency

		for dset_name in datasets:
			data = f[dset_name][:]
			sha256_hash.update(data.tobytes())  # Convert to bytes and hash

		checksum = sha256_hash.hexdigest()
		print(f'{os.path.basename(hdf5_file)}: {checksum}')

	return os.path.basename(hdf5_file), checksum


def compare_checksums(new_csv, comparison_checksum_file):
	'''
	Compares checksums against a known "good" ground truth checksum list
	'''
	print('------------------------------------')
	print(f'Comparing checksums...')
	print('------------------------------------')

	df1 = pd.read_csv(new_csv)
	df2 = pd.read_csv(comparison_checksum_file)

	merged_df = df1.merge(df2, on="file", suffixes=('_original', '_new'))
	mismatches = merged_df[merged_df["checksum_original"] != merged_df["checksum_new"]]

	# Print mismatches
	if not mismatches.empty:
		print("Mismatched files found:")
		print('------------------------------------')
		print(mismatches)
		print('------------------------------------')
	else:
		print("All files have matching checksums.")
		return(True)



if __name__ == "__main__":

	parser = ap.ArgumentParser(description="Generate checksums for HDF5 files in a directory.")
	parser.add_argument('-f', '--comparison_checksum_file', required=False, help='csv file that has comparison checksums')
	parser.add_argument('-i', '--input_directory', required=True, help='Directory containing HDF5 files')
	parser.add_argument('-o', '--output_csv', required=True, help='Output CSV file to save checksums')
	args = parser.parse_args()

	cpus = 12 #hardcoded
	p = multiprocessing.Pool(processes=cpus)

	start_time = time.time()
	async_results = []

	print('------------------------------------')
	print(f'Generating checksums...')
	print('------------------------------------')

	for root, _, files in os.walk(args.input_directory):
		for file in files:
			if file.endswith('.h5'):
				file_path = os.path.join(root, file)
				if cpus > 1:
					async_results.append(p.apply_async(compute_checksum, [file_path]))
				else:
					async_results.append(compute_checksum(file_path))
					
					
	p.close()
	p.join()

	print('------------------------------------')
	print('Collecting results...')

	final_list = []
	for i in async_results:
		sublist = i.get()
		final_list.append(sublist)

	df = pd.DataFrame(final_list, columns=['file', 'checksum'])
	print(df)
	df.to_csv(args.output_csv, index=False)

	print('------------------------------------')
	print(f'Checksums saved to {args.output_csv}')
	print('------------------------------------')
	print(f'Elapsed time:, {round((time.time() - start_time), 2)}s')
	print('------------------------------------')

	if args.comparison_checksum_file is not None:
		compare_checksums(args.output_csv, args.comparison_checksum_file)