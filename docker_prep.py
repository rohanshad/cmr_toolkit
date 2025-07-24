import os
import platform
from pyaml_env import BaseConfig, parse_config

def main():
	device = platform.uname().node.replace('-', '_').lower()
	if 'sh' in device:
		device = 'sherlock'
	elif '211' in device:
		device = 'cubic'
	elif 'rohan' in device:
		device = 'Rohans_MacBook'

	cfg = BaseConfig(parse_config('local_config.yaml'))
	dcm_path = getattr(cfg, device).dcm_benchmark_data
	slack_bot_token = getattr(cfg, "global_settings").slack_bot_token
	slack_bot_channel = getattr(cfg, "global_settings").slack_bot_channel
	num_cpus = getattr(cfg, device).num_cpus

	with open(".env", "w") as f:
		f.write(f'RAW_DICOM_PATH="{dcm_path}"\n')
		f.write(f'SLACK_TOKEN="{slack_bot_token}"\n')
		f.write(f'CHANNEL="{slack_bot_channel}"\n')
		f.write(f'NUM_CPUS="{num_cpus}"\n')

	print(f"Generated .env for device: {device}")

if __name__ == "__main__":
	main()