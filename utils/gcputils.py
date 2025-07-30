'''
Helper functions for gcp blob storage management and dynamic throttling of preprocessing pipelines
Additional utils for mounting gs:buckets to mount points
'''

import os
import random
from google.cloud import storage
import shutil
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from concurrent.futures import ThreadPoolExecutor
import bcolors
from multiprocessing import Process
from collections import defaultdict
import subprocess

DEBUG = False

def wait_if_disk_full(tmp_output_device):
	'''
	Throttling function based on tmp_storage
	'''
	os.makedirs(tmp_output_device, exist_ok=True)
	usage = shutil.disk_usage(tmp_output_device)
	used_ratio = usage.used / usage.total
	while used_ratio > 0.9:
		print(f'Disk usage is {used_ratio:.1%}, throttling pipeline...')
		time.sleep(60)
		usage = shutil.disk_usage(tmp_output_device)
		used_ratio = usage.used / usage.total

@retry(
	retry=retry_if_exception_type(Exception),
	stop=stop_after_attempt(5),
	wait=wait_exponential(multiplier=1, min=2, max=10)
)
def upload_to_gcs(file_path, bucket):
	'''
	Retry if exception with backoff
	Uploads to blob storage 
	'''
	accession = os.path.split(file_path)[1]
	mrn = os.path.split(os.path.dirname(file_path))[1]
	filename = os.path.join(mrn,accession)
	blob = bucket.blob(filename)
	
	try:
		blob.upload_from_filename(file_path, timeout=300)
		print(f'{bcolors.BLUE}Uploaded file: {filename} to {bucket}{bcolors.ENDC}')
		shutil.rmtree(os.path.dirname(file_path))

	except Exception as ex:
		print(f'{bcolors.ERR}ERR: Failed to upload {filename}: {ex}{bcolors.ENDC}')
		raise
	

def gcp_queue_process(queue, path, gcp_dest_bucket):
	'''
	Queue that will populate based on triggers from Handler class
	'''
	print('------------------------------------')
	print(f'{bcolors.BLUE}GCP Upload worker started{bcolors.ENDC}')
	print('------------------------------------')
	while True:
		if not queue.empty():
			h5_filepath = queue.get()
			if h5_filepath is None: 
				print(f'{bcolors.BLUE}GCP Upload worker received stop signal. Exiting.{bcolors.ENDC}')
				break # Terminate process
			else: 
				upload_to_gcs(h5_filepath, gcp_dest_bucket)

class GCP_Upload_Manager:
	'''
	Starts GCP storage client and interfaces with queueing process 
	'''
	def __init__(self, path, upload_queue, gcp_dest_bucket):

		self.path = path
		self.queue = upload_queue
		self.bucket_name = gcp_dest_bucket
		self.process = Process(target=self.run, daemon=False)

	def run(self):
		client = storage.Client()
		bucket = client.bucket(self.bucket_name)
		gcp_queue_process(self.queue, self.path, bucket)

	def start(self):
		self.process.start()

	def wait_until_done(self):
		self.process.join()

def mount_gcs_bucket(bucket_name: str, mount_point: str, read_only: bool = True, implicit_dirs: bool = True) -> bool:
	'''
	Mounts a Google Cloud Storage bucket to a local filesystem mount point using gcsfuse.
	ONLY WORKS WITH LINUX

	Args:
		bucket_name (str): The name of the GCS bucket to mount (e.g., 'gs:something_something').
		mount_point (str): Local mount point (e.g. '/mnt/bucket')
		read_only (bool): If True, mount the bucket as read-only. Defaults to True.
		implicit_dirs (bool): If True, enable implicit directories. This is often
							  necessary for buckets that don't have explicit directory
							  objects. Defaults to True.

	Returns:
		bool: True if the bucket was successfully mounted, False otherwise.
	'''
	
	bucket_name = bucket_name[3:]
	  
	# Check if the mount point is already mounted
	if os.path.ismount(mount_point):
		print(f'{bcolors.WARN}Mount point "{mount_point}" is already mounted. Assuming its correctly mounted.{bcolors.ENDC}')
		return True

	os.makedirs(mount_point, exist_ok=True)

	# Build the gcsfuse command
	command = ['gcsfuse']

	if read_only:
		command.append('-o')
		command.append('ro') # Read-only option

	if implicit_dirs:
		command.append('--implicit-dirs') # Treat object prefixes as directories

	command.extend([bucket_name, mount_point])

	print(f'Attempting to mount GCS bucket "{bucket_name}" to "{mount_point}"...')
	print(f'Command: {" ".join(command)}')

	try:
		# Run the command, capture output for debugging
		# check=True will raise CalledProcessError if the command returns a non-zero exit code
		result = subprocess.run(command, check=True, capture_output=True, text=True)
		print('Mount command output:')
		print(result.stdout)
		if result.stderr:
			print('Mount command error output:')
			print(result.stderr)
		print(f'{bcolors.OK}Successfully mounted "{bucket_name}" to "{mount_point}".{bcolors.ENDC}')
		return True
	except FileNotFoundError:
		print(f'{bcolors.ERR}Error: "gcsfuse" command not found. Please ensure gcsfuse is installed and in your systems PATH.{bcolors.ENDC}')
		print('Installation instructions: https://cloud.google.com/storage/docs/cloud-storage-fuse')
		return False
	except subprocess.CalledProcessError as e:
		print(f'{bcolors.ERR}Error mounting bucket: Command failed with exit code {e.returncode}{bcolors.ENDC}')
		print(f'STDOUT: {e.stdout}')
		print(f'STDERR: {e.stderr}')
		print('Please check gcsfuse logs, permissions, and bucket name.')
		return False
	except Exception as e:
		print(f'{bcolors.ERR}An unexpected error occurred during mounting: {e}{bcolors.ENDC}')
		return False

def unmount_gcs_bucket(mount_point: str) -> bool:
	'''
	Unmounts a GCS FUSE mounted directory.
	ONLY WORKS WITH LINUX

	Args:
		mount_point (str): The local directory path where the bucket is mounted.

	Returns:
		bool: True if the bucket was successfully unmounted, False otherwise.
	'''
	if not os.path.ismount(mount_point):
		print(f'"{mount_point}" is not a mount point or is not currently mounted.')
		return False

	# Use fusermount -u for FUSE filesystems on Linux.
	# On macOS, use diskutil unmount or umount.
	if sys.platform.startswith('linux'):
		command = ['fusermount', '-u', mount_point]
	elif sys.platform == 'darwin': # macOS
		command = ['umount', mount_point]
	else:
		print(f'Unmounting is not directly supported on this OS: {sys.platform}')
		return False

	print(f'Attempting to unmount "{mount_point}"...')
	print(f'Command: {" ".join(command)}')

	try:
		result = subprocess.run(command, check=True, capture_output=True, text=True)
		print('Unmount command output:')
		print(result.stdout)
		if result.stderr:
			print('Unmount command error output:')
			print(result.stderr)
		print(f'Successfully unmounted "{mount_point}".')
		return True

	except FileNotFoundError:
		print(f'Error: Unmount command ("{command[0]}"") not found.'
			  'Ensure fusermount (Linux) or umount (macOS) is available.')
		return False

	except subprocess.CalledProcessError as e:
		print(f'Error unmounting bucket: Command failed with exit code {e.returncode}')
		print(f'STDOUT: {e.stdout}')
		print(f'STDERR: {e.stderr}')
		print('You might need to manually unmount (e.g., `sudo umount /path/to/mount`).')
		return False
		
	except Exception as e:
		print(f'An unexpected error occurred during unmounting: {e}')
		return False


