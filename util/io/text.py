import sys

sys.path.insert(0, "..")
import json
import yaml
from util.log_utils import get_logger

log = get_logger(__name__)


def load_json(path: str, silent=True):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not silent:
        log.info(f"load {path}")
    return obj


def save_json(obj, path: str, silent=True, sort_keys=True):
    with open(path, "w", encoding="utf-8") as f:
        # https://github.com/IsaacChanghau/VSLNet/blob/master/util/data_util.py
        f.write(json.dumps(obj, indent=4, sort_keys=sort_keys))
    if not silent:
        log.info(f"saved to {path}")


def save_yaml(obj, path: str, silent=True):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(obj, f)
    if not silent:
        log.info(f"saved to {path}")
