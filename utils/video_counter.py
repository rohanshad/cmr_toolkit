import h5py 
import os 
import glob
from tqdm import tqdm
import argparse as ap

def count_unique_videos(hdf5_file):
	'''
	Counts the number of unique cine-sequence videos for each anatomical slice
	'''
	df = h5py.File(hdf5_file, 'r')
	views = list(df.keys())

	counter = 0
	for i in views:
		if len(df[i].attrs['slice_frames']) == 0:
			counter = counter + 1
		else:
			counter = counter + len(df[i].attrs['slice_frames'])

	return counter

def frame_counter(hdf5_file):
	'''
	Self explanatory, counts frames 
	'''
	df = h5p.File(hdf5_file, 'r')
	views = list(df.keys())

	for i in views:
		if len(df[i].attrs['slice_frames']) == 0:
			frames = df[i].shape[1]
		else:
			frames = slice_frames[1]-slice_frams[0]

		print(frames)

if __name__ == '__main__':
	parser = ap.ArgumentParser(
		description="Counts unique videos in a hdf cMRI data directory",
		epilog="Version 1.1; Created by Rohan Shad, MD"
	)

	parser.add_argument('-r', '--data_dir', metavar='', required=False, help='Full path to root directory')	
	args = vars(parser.parse_args())
	root_dir = args['data_dir']

	start_time = time.time()

	video_count = 0

	filelist = glob.glob(os.path.join(root_dir, '*', '*'))
	for file in tqdm(filelist):
		video_count = video_count + count_unique_videos(file)


	print(f'Total number of unique videos available: {video_count}')
	print(f'Elapsed time: {round((time.time() - start_time), 2)}')