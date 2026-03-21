import os
from utils.local_config import get_cfg, get_global_cfg, get_profile_name

def main():
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