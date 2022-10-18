'''
Reads in master hdf5 storage that is the output of preprocess_mri.py and builds view-specific hdf5 datasets
for faster train-time performance. Uses the same csv safelist method of generating specific datasets if needed

Input hdf5 dataset format:
stanford_RF3da3244
├── RF3da3581.h5
├── RF3lv2173.h5
	├── 4CH_FIESTA_BH 			{data: 4d array} {attr: fps, total images}
	├── SAX_FIESTA_BH_1			{data: 4d array} {attr: fps, total images, slice frame index}
	├── SAX_FIESTA_BH_2			{data: 4d array} {attr: fps, total images, slice frame index}
	├── STACK_LV3CH_FIESTA_BH 	{data: 4d array} {attr: fps, total images, slice frame index}


Output hdf5 format:
stanford_RF3da3244
├── RF3da3581.h5
├── RF3lv2173.h5
	├── 4CH 		{data: 4d array} {attr: fps, total images}
	├── SAX			{data: 4d array} {attr: fps, total images, slice frame index}
	├── 3CH			{data: 4d array} {attr: fps, total images}

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
import matplotlib.pyplot as plt
import time



def extract_series(root_dir, folder_name, output_dir, series, series_map, compression):
	'''
	Opens a folder [name: institution_mrn] containing hdf5 datasets with filename [name = accession number]
	series_df is a pandas dataframe that matches raw series names to the cleaned versions from series_map


	Series of interest copied into separate hdf5 with all attrs. Series name at this point is cleaned version.
	Function does not return anything
	'''

	accession_list = glob.glob(os.path.join(root_dir,folder_name,'*'))
	series = [series] if isinstance(series, str) else series
	safe_makedir(os.path.join(output_dir, folder_name))

	for accession in accession_list:

		# Maps out the raw series name to the cleaned naming scheme for consistency
		source_hdf5 = h5py.File(os.path.join(root_dir, folder_name, accession), 'r')
		raw_series_names = list(source_hdf5.keys())
		series_df = series_map[series_map['series_description'].isin(raw_series_names)]

		for s in series:
			try:
				# Don't know why this was giving problems on Sherlock for a while.
				raw_series_name = series_df.loc[(series_df['cleaned_series_names'] == s)].values[0,0]
				print('Extracting', s, 'for', folder_name, '...')
				print(series_df.loc[(series_df['cleaned_series_names'] == s)])
				dest_hdf5 = h5py.File(os.path.join(output_dir, folder_name, os.path.basename(accession)), 'a')

				# Create new hdf5 file
				source_hdf5.copy(source_hdf5[raw_series_name], dest_hdf5, name=s)
				dest_hdf5.close()

			except:
				print('Could not extract', s, 'for', folder_name+'-'+os.path.basename(accession))


def safe_makedir(path):
	'''
	Nick's little safe mkdir function
	'''
	if not os.path.exists(path):
		os.makedirs(path)


if __name__ == '__main__':

	parser = ap.ArgumentParser(
		description="Build ML ready hdf5 dataset",
		epilog="Version 0.1; Created by Rohan Shad, MD"
	)

	parser.add_argument('-r', '--root_dir', metavar='', required=False, help='Full path to root directory', default='/Users/rohanshad/PHI Safe/dataset_build_test/preprocessed_data')
	parser.add_argument('-l', '--csv_list', metavar='', required=False, help='Process only files listed in csv_list.csv', default=None)
	parser.add_argument('-o', '--output_dir', metavar='', required=False, help='Where all output files will be stored', default='/Users/rohanshad/PHI Safe/dataset_build_test/allviews')
	parser.add_argument('-z', '--compression', metavar='', required=False, help='Compression type (gzip pr lzf)', default='gzip')
	parser.add_argument('-c', '--cpus', metavar='', type=int, default='1', help='number of cores to use in multiprocessing')
	parser.add_argument('-d', '--debug', action='store_true', default=False)
	parser.add_argument('-s', '--series', metavar='', type=str, default='4CH', nargs="*", help='MRI series to process')
	parser.add_argument('-v', '--visualize', action='store_true', required=False, help='print data from random hdf5 file in output folder')
	parser.add_argument('-m', '--mapping_csv', metavar='', required=True, help='')
	args = vars(parser.parse_args())
	print(args)

	root_dir = args['root_dir']
	csv_list = args['csv_list']
	compression = args['compression']
	cpus = args['cpus']
	visualize = args['visualize']
	debug = args['debug']
	mapping_csv = args['mapping_csv']
	series = args['series']

	#For gcloud:
	output_dir = args['output_dir']
	safe_makedir(output_dir)

	#### Visualize one frame from hdf5 MRI array (might keep this in a utils script later) ####
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

			print('------------------------------------')
			print('Plotting', os.path.basename(random_file))
			dat = h5py.File(os.path.join(output_dir, random_file), 'r')
			print('Avaialble views:', list(dat.keys()))

			#Reading hdf5 file once is faster when you have to open multiple arrays from it afterwards (I/O bound)
			dat = dat.get(random.choice(list(dat.keys())))
			print(dat)

			# Plotting Code (hddf5 is saved as f, c, h, w)
			array = np.array(dat).transpose(0, 2, 3, 1)
			print(f'number of frames: {np.size(array, 1)}')
			plt.imshow((array[random.choice(list(range(np.size(array, 0))))])[:,:,1]/255, cmap='magma')
			plt.show()


	### Debugger Module ####
	elif debug == True:

		input_filelist = glob.glob(os.path.join(root_dir,'*','*'))
		processed_filelist = glob.glob(os.path.join(output_dir,'*','*'))

		print()
		print('Running in debug mode...')

		print('------------------------------------')
		print('Total scans (Accession numbers) available:', len(input_filelist))
		print('Checking failure rate...')
		print('Total scans (Accession numbers) processed:', len(processed_filelist))

		main_view_list = []
		outputs = []
		for f in processed_filelist:
			df = h5py.File(f, 'r')
			views = list(df.keys())

			parent_folder = f.split('/')[-2]
			accession = f.split('/')[-1]
			for v in views:
				view_list = [parent_folder, accession, v]

			main_view_list.append(view_list)
			outputs.append(parent_folder+'-'+accession)

		print('Total number of views saved:', len(sum(main_view_list, [])))

		inputs = []
		for i in input_filelist:
			parent_folder = i.split('/')[-2]
			accession = i.split('/')[-1]

			inputs.append(parent_folder+'-'+accession)

		incomplete = set(inputs).difference(set(outputs))
		print('Did not process', len(incomplete), 'files.')
		print('Exporting failed files to csv')
		incomplete_df = pd.DataFrame(list(incomplete), columns = ["filenames"])
		incomplete_df.to_csv('failed_to_process.csv', index=False)
		print('------------------------------------')


	else:
		# Read series name map #
		series_map = pd.read_csv(os.path.join(root_dir, mapping_csv))
		# Main run command to convert dcm files to hdf5
		p = multiprocessing.Pool(processes=cpus)

		if csv_list is not None:
			try:
				df = pd.read_csv(os.path.join(root_dir, csv_list))
				print(df)
				filenames = df['filenames'].tolist()

				files_in_dir = os.listdir(root_dir)
				folder_names = set(filenames.split('-')[0]).intersection(files_in_dir)
			except:
			 	print('Could not open csv safelist')

		else:
			folder_names = os.listdir(root_dir)

		start_time = time.time()
		print('Extracting all', series, 'views:')
		for f in folder_names:
			# Each folder is a unique MRN with hdf5 files inside that represent accession numbers

			if cpus > 1:
				p.apply_async(extract_series,[root_dir, f, output_dir, series, series_map, compression])
			else:
				extract_series(root_dir, f, output_dir, series, series_map, compression)

		p.close()
		p.join()

		# Deletes empty directories in output
		for out_dir in os.listdir(output_dir):
			if len(os.listdir(os.path.join(output_dir,out_dir))) == 0:
				rmtree(os.path.join(output_dir,out_dir))


		print('------------------------------------')

		print('Elapsed time:', round((time.time() - start_time), 2))
		print('Total input MRNs:', len(folder_names))
		print('Total input Accessions:', len(glob.glob(os.path.join(root_dir,'*','*'))))
		print('Total output Accessions:', len(glob.glob(os.path.join(output_dir,'*','*'))))

		print(('------------------------------------'))
