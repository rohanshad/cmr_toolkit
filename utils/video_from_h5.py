'''
Exports hdf5 datastores to mp4 and jpg in a neat nested folder structure
stanford_RF3da2ty4 (MRN parent folder)
├── RFasfe3581 (Accession subdir)
├── RFxlv2173
	├── 4CH_FIESTA_BH.mp4
	├── SAX_FIESTA_BH_1.mp4
	├── SAX_FIESTA_BH_2.mp4			
	├── STACK_LV3CH_FIESTA_BH.mp4
'''
import h5py 
import os
import ffmpeg
import multiprocessing
import argparse as ap
import glob
import time
import pandas as pd
import numpy as np
import bcolors

def ffmpeg_writer(hdf5_filepath, series, output_dir, channels):
	mrn = os.path.split(os.path.dirname(hdf5_filepath))[1]
	accession = os.path.basename(hdf5_filepath)[:-3]

	try:
		with h5py.File(hdf5_filepath, 'r') as file:

			arr = file[series][()]  # shape: [c,f,h.w] for old processed runs, new are [f,c,h,w]
			if channels == 'grey':
				assert len(arr.shape) < 4, "Channel check failed for channels = greyscale"
				
				### NP.REPEAT TO 3 CHANNEL SHAPE ###
				if len(arr.shape) == 3:
					arr = np.repeat(arr[:,None,:,:],repeats=3,axis=1) # [f, c=3, h, w]
				elif len(arr.shape) == 2:
					arr = np.repeat(arr[None,:,:],repeats=3,axis=0) # [c=3, h, w]

			elif channels == 'rgb':
				assert arr.shape[-3] == 3, "Channel check failed for channels = rgb"

			if len(arr.shape) == 4:
				arr = arr.transpose(0, 2, 3, 1) # Transpose to (f, h, w, c)
			elif len(arr.shape) == 3:
				arr = arr.transpose(1, 2, 0) # Transpose to (h, w, c)

			# Normalize globally
			vmin, vmax = np.min(arr), np.max(arr)
			vid = np.clip((arr - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)

			#vid = np.array(file[series], dtype=np.uint8).transpose(1,2,3,0)
			os.makedirs(os.path.join(output_dir,mrn,accession), exist_ok=True)

			process = (
				ffmpeg
				.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'360,360')
				.filter('normalize',strength=1)
				.output(f'{os.path.join(output_dir,mrn,accession,series+".mp4")}', pix_fmt='yuv420p', crf='14', r=24, loglevel="quiet")
				.overwrite_output()
				.run_async(pipe_stdin=True)
				)

			for frame in vid:
				process.stdin.write(frame.tobytes())

			process.stdin.close()
			process.wait()

			print(f'{bcolors.BLUE}{mrn}-{accession}{bcolors.ENDC} exported series: {series}')

	except KeyboardInterrupt:
		process.stdin.close()
		process.wait()

def catalog_h5_keys(input_dir):

	series = []
	filepaths = []

	filelist = glob.glob(os.path.join(input_dir,'*','*h5'))
	for file in filelist:
		with h5py.File(file, 'r') as df:
			series_list = list(df.keys())
			for s in series_list:
				series.append(s)
				filepaths.append(file)

	df = pd.DataFrame.from_dict({'filepaths':filepaths, 'series':series})
	return df


if __name__ == "__main__":

	parser = ap.ArgumentParser(description="Generates mp4 video and jpg for troubleshooting / review of hdf5 arrays.")
	parser.add_argument('-i', '--input_dir', required=True, help='Directory containing HDF5 files')
	parser.add_argument('-o', '--output_dir', required=True, help='Output path, subdirs for outputs will be generated here')
	parser.add_argument('-c', '--cpus', required=True, default=12, type=int, help="Number of CPUs")
	parser.add_argument('--channels', required=True, default='grey', type=str, help="Choice between 'grey' for greyscale stored hdf5 and 'rgb'")
	args = parser.parse_args()

	p = multiprocessing.Pool(processes=args.cpus)

	start_time = time.time()
	async_results = []

	df = catalog_h5_keys(args.input_dir)

	for index, row in df.iterrows():
		if args.cpus > 1:
			p.apply_async(ffmpeg_writer(row['filepaths'], row['series'], args.output_dir, args.channels))
		else:
			ffmpeg_writer(row['filepaths'], row['series'], args.output_dir, args.channels)

	p.close()
	p.join()

	print('------------------------------------')

	print('Elapsed time:', round((time.time() - start_time), 2))
	print('Total input Accessions:', len(df['filepaths'].unique()))
	print('Total series processed:', len(glob.glob(os.path.join(args.output_dir,'*','*',"*"))))
	print(('------------------------------------'))
		


