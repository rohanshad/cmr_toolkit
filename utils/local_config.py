import os
import platform
from pathlib import Path
from pyaml_env import BaseConfig, parse_config

_CONFIG_PATH = Path(__file__).resolve().parent.parent / 'local_config.yaml'


def _resolve_profile(raw: dict) -> str:
	'''
	Resolve the current hostname to a local_config.yaml profile name.
	Checks DEVICE_NAME env var first (allows Docker/CI override),
	then hostname_patterns (substring match, first match wins),
	then falls back to exact hostname as the profile key.
	'''
	device_name = os.environ.get('DEVICE_NAME', '').strip()
	if device_name:
		return device_name
	hostname = platform.uname().node.replace('-', '_')
	for pattern, name in raw.get('hostname_patterns', {}).items():
		if pattern in hostname:
			return name
	return hostname


def get_cfg():
	'''
	Return the device-specific config block from local_config.yaml.
	Raises RuntimeError if no matching profile is found.
	'''
	raw = parse_config(_CONFIG_PATH)
	profile = _resolve_profile(raw)
	cfg = BaseConfig(raw)
	try:
		return getattr(cfg, profile)
	except AttributeError:
		raise RuntimeError(
			f"No config profile found for hostname "
			f"'{platform.uname().node}' (resolved to '{profile}'). "
			f"Add it to local_config.yaml."
		)


def get_global_cfg():
	'''
	Return the global_settings block from local_config.yaml.
	'''
	raw = parse_config(_CONFIG_PATH)
	return BaseConfig(raw).global_settings


def get_profile_name() -> str:
	'''
	Return the resolved profile name for the current host (e.g. 'sherlock', 'parcc').
	Useful when profile-keyed logic is needed outside of path lookups.
	'''
	raw = parse_config(_CONFIG_PATH)
	return _resolve_profile(raw)
