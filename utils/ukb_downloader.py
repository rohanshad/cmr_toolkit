import os 
import time
import numpy as np
import argparse as ap
import glob
import subprocess 
import multiprocessing
import pandas as pd

class BulkDownloaderUKBB:
	'''
	Wrapper for chunked, parallel downloading of UK Biobank bulk imaging data
	using the `ukbfetch` command-line utility.

	This class reads a UK Biobank `.bulk` file, computes 1-indexed starting
	row offsets corresponding to fixed-size chunks (default: 1000 rows),
	and invokes `ukbfetch` repeatedly to download data in manageable batches.

	The design mirrors the native `ukbfetch` semantics:
	  - `-s` specifies the starting row (1-indexed)
	  - `-m` specifies the number of rows to process per call

	Typical usage is to generate a list of starting rows via `chunk_bulkfile()`
	and dispatch `ukbfetch` calls sequentially or in parallel (e.g. via
	`multiprocessing.Pool`) to enable robust, restartable bulk downloads.

	This class does not manage authentication state beyond passing the
	provided keyfile to each `ukbfetch` invocation; each call is treated as
	independent.

	ukbfetch_exec : (str) Full path to the `ukbfetch` executable.
	bulkfile : (str) Path to the UK Biobank `.bulk` file listing data items to download.
	keyfile : (str) Path to the UK Biobank authentication key file.
	'''
	
	def __init__(self, ukbfetch_exec, bulkfile, keyfile):
		self.ukbfetch_exec = ukbfetch_exec
		self.bulkfile = bulkfile
		self.keyfile = keyfile


	def chunk_bulkfile(self):
		df = pd.read_table(self.bulkfile, sep=" ", header=None, names=["eid", "data_item"])
		chunk_size = 1000
		chunked_dfs = [
		    df.iloc[i:i + chunk_size]
		    for i in range(0, len(df), chunk_size)
		]

		# Sanity checks:
		print('------------------------------------')
		print(f'Number of chunks: {len(chunked_dfs)}')
		print(f'Rows in first chunk: {chunked_dfs[0].shape}')
		print(f'Rows in last chunk: {chunked_dfs[-1].shape}')
		print('------------------------------------')

		start_rows = list(range(1, len(df) + 1, chunk_size))
		return start_rows


	def ukbfetch(self, start):
		'''
		Downloads 1000 bulk_data items at at time 
		'''
		p = subprocess.Popen([str(self.ukbfetch_exec), str('-b'+self.bulkfile), str('-s'+str(start)), str('-m1000'), str('-a'+self.keyfile)])
		p.wait()




if __name__ == "__main__":
	parser = ap.ArgumentParser(
		description="Bulk UKB data downloading wrapper for ukbfetch",
		epilog="Version 2.0; Created by Rohan Shad, MD"
	)

	parser.add_argument('-e', '--ukbfetch_exec', metavar='', required=True, help='Full path to ukbfetch executable', default='/mnt/scratch/ukbiobank_prep/ukbfetch')
	parser.add_argument('-o', '--output_dir', metavar='', required=True, help='Path to output directory')
	parser.add_argument('-c', '--cpus', metavar='', type=int, default='20',help='Num parallel connections')
	parser.add_argument('-b', '--bulkfile', metavar='', required=True, help='Full path to bulkfile.bulk')
	parser.add_argument('-a', '--keyfile', metavar='', required=True, help='Full path to auth.key file')


	args = vars(parser.parse_args())


	os.makedirs(args['output_dir'], exist_ok=True)
	os.chdir(args['output_dir'])

	start_time = time.time()
	p = multiprocessing.Pool(processes=args['cpus'])

	bulk_downloader = BulkDownloaderUKBB(ukbfetch_exec = args['ukbfetch_exec'], bulkfile = args['bulkfile'], keyfile = args['keyfile'])
	start_rows = bulk_downloader.chunk_bulkfile()

	for s in start_rows:
		if args['cpus'] > 1:
			p.apply_async(bulk_downloader.ukbfetch, [s])
		else:
			bulk_downloader.ukbfetch(s)

	p.close()
	p.join()

	print('------------------------------------')
	print(f'Elapsed time: {round((time.time() - start_time), 2)}s')
	print('------------------------------------')

