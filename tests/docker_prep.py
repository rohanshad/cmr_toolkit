'''
docker_prep.py — Generate .env file from local_config.yaml for Docker test pipeline.

Reads the device-specific config block from local_config.yaml via local_config.py
(resolved by hostname or DEVICE_NAME env var) and writes a .env file at the repo root
containing the variables expected by docker-compose.yml:

    RAW_DICOM_PATH  — path to benchmark DICOM test cases
    SLACK_TOKEN     — Slack bot token for pipeline notifications
    CHANNEL         — Slack channel ID
    NUM_CPUS        — CPU count for multiprocessing inside the container
    TMP_DIR         — scratch directory for intermediate extraction

Run automatically by tests/run_docker_tests.sh before docker compose up.
Can also be run manually to regenerate .env after changing local_config.yaml.

Usage:
    python tests/docker_prep.py
'''

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.local_config import get_cfg, get_global_cfg, get_profile_name

def main():
	'''
	Read local_config.yaml and write a .env file at the repo root.

	Resolves the current device profile via get_cfg() (hostname matching or
	DEVICE_NAME env var), reads the global settings block for credentials,
	and writes all required docker-compose environment variables to .env.
	Prints the resolved profile name on success.
	'''
	_cfg    = get_cfg()
	_global = get_global_cfg()
	dcm_path          = _cfg.dcm_benchmark_data
	slack_bot_token   = _global.slack_bot_token
	slack_bot_channel = _global.slack_bot_channel
	num_cpus          = _cfg.num_cpus
	tmp_dir           = _cfg.tmp_dir

	with open(".env", "w") as f:
		f.write(f'RAW_DICOM_PATH="{dcm_path}"\n')
		f.write(f'SLACK_TOKEN="{slack_bot_token}"\n')
		f.write(f'CHANNEL="{slack_bot_channel}"\n')
		f.write(f'NUM_CPUS="{num_cpus}"\n')
		f.write(f'TMP_DIR="{tmp_dir}"\n')

	print(f"Generated .env for device: {get_profile_name()}")

if __name__ == "__main__":
	main()