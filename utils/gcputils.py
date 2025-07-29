'''
Helper functions for gcp blob storage management and dynamic throttling of preprocessing pipelines
'''

import random
from google.cloud import storage
import shutil
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor


def wait_if_disk_full(tmp_output_device):
	'''
	Throttling function based on tmp_storage
	'''
	usage = shutil.disk_usage(tmp_output_device)
	used_ratio = usage.used / usage.total
	while used_ratio > 0.9:
		print(f'Disk usage is {used_ratio:.1%}, throttling pipeline...')
		time.sleep(60)
		usage = shutil.disk_usage(tmp_output_device)
		used_ratio = usage.used / usage.total
	
	#print(f'Disk usage OK: {used_ratio:.1%}, continuing.')


@retry(
	retry=retry_if_exception_type(Exception),
	stop=stop_after_attempt(5),
	wait=wait_exponential(multiplier=1, min=2, max=10)
)
def upload_to_gcs(file_path, storage_client, bucket):
	'''
	Retry if exception with backoff
	Uploads to blob storage 
	'''

	filename = os.basepath(file_path)
	blob = bucket.blob(filename)
	try:
		blob.upload_from_filename(filename)
		print(f'Uploaded file: {filename} to {bucket}')
	except Exception as ex:
		print(f'ERR: Failed to upload.')

	os.remove(filename)



def wait_for_file_complete(file_path, timeout=120):
	'''
	Wait until the file size has stabilized, or until timeout (sec).
	'''
	start_time = time.time()
	prev_size = -1

	while time.time() - start_time < timeout:
		try:
			curr_size = os.path.getsize(file_path)
		except FileNotFoundError:
			time.sleep(1)
			continue

		if curr_size == prev_size:
			# File hasn't changed size for STABILIZATION_SECONDS (Set to 120s for now)
			stable_start = time.time()
			while time.time() - stable_start < 120:
				time.sleep(1)
				new_size = os.path.getsize(file_path)
				if new_size != curr_size:
					prev_size = new_size
					break
			else:
				return True  # Size stable for STABILIZATION_SECONDS (120s)
		else:
			prev_size = curr_size

		time.sleep(1)

	raise TimeoutError(f'File {file_path} did not stabilize in time.')

class NewFileHandler(FileSystemEventHandler):
	def __init__(self, executor, output_dir):
		self.executor = executor
		self.storage_client = storage.Client()
		self.bucket = client.bucket(output_dir)

	def on_created(self, event):
		if event.is_directory or not event.src_path.endswith('.h5'):
			return
		file_path = event.src_path
		self.executor.submit(self._safe_upload, file_path)

	def _safe_upload(self, file_path):
		try:
			wait_for_file_complete(file_path)
			upload_file_gcs(file_path, self.storage_client, self.bucket)

		except Exception as ex:
			print(f'WARN: Could not upload blob: {filename}')
			print(ex)


def start_watcher(output_dir):	
	'''
	Only does something if output_dir passed into this is a gs:bucket
	'''
	if output_dir[3:] == "gs:":	
		executor = ThreadPoolExecutor(max_workers=4)
		handler = NewFileHandler(executor, output_dir)
		observer = Observer()
		observer.schedule(handler, output_dir, recursive=False)
		observer.start()

		print(f'Watching {output_dir} for new files...')
		try:
			while True:
				time.sleep(1)
		except KeyboardInterrupt:
			observer.stop()
		observer.join()

	else:
		pass

