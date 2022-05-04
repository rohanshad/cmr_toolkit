'''
Script to visualize and plot MRI hdf5 dataset objects 
'''

import torch
import matplotlib
import matplotlib.pyplot as plt
import random
import os 
import numpy as np
import h5py
import argparse as ap
import glob
import imageio
import imageio_ffmpeg 
import subprocess

from pyaml_env import BaseConfig, parse_config

# Read local_config.yaml for local variables 
cfg = BaseConfig(parse_config('../local_config.yaml'))
TMP_DIR = cfg.tmp_dir
BUCKET_NAME = cfg.bucket_name

def generate_video_mp4(array, view, root_dir, input_hdf5, output_directory):
	'''
	Generates a mp4 video from input arrays using ffmpeg
	Retains magma color scheme because it looks dope, but might want to use b/w instead
	'''
	
	save_dir = os.path.join(root_dir, 'tmp', os.path.basename(input_hdf5[:-3])+'_pngs')
	os.makedirs(save_dir, exist_ok = True)

	for i in range(array.shape[0]):
		plt.imshow(array[i][:,:,1]/225, cmap='magma')
		plt.axis('off')
		# Might keep axis but remove bbox 
		plt.savefig(os.path.join(save_dir, 'frame%02d.png' %i), bbox_inches='tight')	

	output_filename = os.path.join(output_directory, os.path.basename(input_hdf5)[:-3]+'_'+view+'.mp4')
	input_filename = os.path.join(root_dir, 'tmp', os.path.basename(input_hdf5[:-3])+'_pngs','frame%02d.png')

	command = 'ffmpeg -i {input_filename} ' \
				'-loglevel error -c:v libx264 -c:a copy ' \
				'-vf scale=380:-2 '\
				'-r 20 -pix_fmt yuv420p ' \
				'"{output_filename}"'.format(
					input_filename=input_filename,
					output_filename=output_filename
				)
	print(command)
	subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)


def hdf5list_to_array(input_hdf5_list, series):
	'''
	Processes a list of random hdf5 files for a specific series and returns a list of np.arrays
	'''

	array_list = []
	for input_hdf5 in input_hdf5_list:
		try:
			dat = h5py.File(input_hdf5, 'r')[series]
			array = np.array(dat).transpose(1, 2, 3, 0)
			array_list.append(array)
		except:
			print('No', series, 'found, skipping...')

	return array_list

def hdf5_to_array(input_hdf5, series):
	'''
	Processes a random hdf5 file of a specific series and returns a np.array
	'''
	dat = h5py.File(input_hdf5, 'r')[series]
	array = np.array(dat).transpose(1, 2, 3, 0)

	return array
	
def hdf5stack_to_array(input_hdf5, series):
	'''
	Processes a random hdf5 file of a specific series and returns a np.array and list of frame slices
	'''
	dat = h5py.File(input_hdf5, 'r')[series]
	array = np.array(dat).transpose(1, 2, 3, 0)
	print(array.shape)
	slices = dat.attrs['slice_frames']

	return array, slices
	
def plot_random_frames(input_arrays):
	'''
	Plots on random frame each from list of input arrays
	
	Args:
		array (float32):	np.array from hdf5 dataset {float32}
	'''

	img_list = []

	for array in input_arrays:
		random_frame = np.random.choice(np.size(array, 0))
		img = array[random_frame]
		img_list.append(img)

	# Plotting code
	_, axs = plt.subplots(3, 5, figsize=(6, 6))
	axs = axs.flatten()
	for img, ax in zip(img_list, axs):
		ax.imshow(img[:,:,1]/225, cmap='viridis')
	plt.show()


def plot_study_sequence(array, skip_period):
	'''
	Plots each frame of a single video / MRI slice on time axis
	
	Args:
		array (float32):	np.array from hdf5 dataset
		skip_period	(int):	user supplied skip period for plotting
	'''

	# Array input is f, c, h, w
	print(f'Total number of frames: {np.size(array, 0)}')
	total_frames = np.size(array, 0)
	array = array[0:total_frames:skip_period,:,:,:]

	img_list = []

	for i in range(array.shape[0]):
		img = array[i]
		img_list.append(img)

	# Plotting code
	_, axs = plt.subplots(1, 5, figsize=(10, 3))
	axs = axs.flatten()
	for img, ax in zip(img_list, axs):
		ax.imshow(img[:,:,1]/225, cmap='magma')
	plt.show()


def plot_study_slices(array, slice_index, skip_period):
	'''
	Plots the first frame of each new slice in a stack (eg. SAX stack)
	
	Args:
		array (float32):	np.array from hdf5 dataset
		slice_index (int):	list of integers from hdf5.attribute 'slice_frames'
		skip_period	(int):	user supplied skip period for plotting
	'''

	total_frames = np.size(array, 0)
	slice_index = slice_index[::skip_period]
	print(slice_index)
	img_list = []

	# Note that the first 'slice index' is where it first changes.
	# Video #1 is thus array[0:i]
	for i in slice_index:
		img = array[i]
		img_list.append(img)

	# Plotting code
	_, axs = plt.subplots(1, 5, figsize=(10, 2))
	axs = axs.flatten()
	for img, ax in zip(img_list, axs):
		ax.imshow(img[:,:,1]/225, cmap='magma')
	plt.show()


if __name__ == '__main__':
	

	parser = ap.ArgumentParser(
		description="Plot samples from MRI hdf5 datasets v0.1",
		epilog="Version 0.1; Created by Rohan Shad, MD"
	)

	parser.add_argument('-r', '--root_dir', metavar='', required=False, help='Full path to root directory', default='/Users/rohanshad/PHI Safe/test_mri_downloads/output')
	parser.add_argument('-v', '--view_name', metavar='', required=False, default='4CH_FIESTA_BH', help='View to plot')
	parser.add_argument('-s', '--skip_period', metavar='', type=int, default=0, help='skip interval for some plots')
	parser.add_argument('-m', '--mode', metavar='', required=True, help='options: grid, sequence, slices, video')
	
	# Add some shit here for specific resolution shit

	args = vars(parser.parse_args())
	#print(args)


	root_dir = args['root_dir']
	view_name = args['view_name']
	skip_period = args['skip_period'] # this should be int
	mode = args['mode']

	filenames =  glob.glob(os.path.join(root_dir, '*', '*'))
	file_list_final = []

	for f in filenames:
		if ".h5" in f:
			file_list_final.append(f)


	if mode == 'grid':
		random_file = random.sample(file_list_final, 15)

		# Plots a grid of randomly selected files 

		plot_random_frames(hdf5list_to_array(random_file, view_name))

	elif mode == 'sequence':
		random_file = random.choice(file_list_final)

		# Plots a long sequence of images for cardiac cycle
		plot_study_sequence(hdf5_to_array(random_file, view_name), skip_period = 4)

	elif mode == 'slices':
		random_file = random.choice(file_list_final)

		# Plots a sequence of stacks for a single short axis series
		print('Plotting', view_name, 'for', random_file)
		plot_study_slices(*hdf5stack_to_array(random_file, view_name), skip_period = 2)


	elif mode == 'video':
		'''
		TODO: Multiprocess this later
		'''
		random.shuffle(file_list_final)
		for f in file_list_final:
			try:
				arr = hdf5_to_array(f, view_name)
				generate_video_mp4(arr, view_name, root_dir, f, TMP_DIR)
			except Exception as e:
				print(e)
				continue







