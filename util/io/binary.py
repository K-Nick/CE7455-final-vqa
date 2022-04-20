import sys

sys.path.insert(0, "..")
import joblib
import pickle
from util.log_utils import get_logger

log = get_logger(__name__)


def load_pickle(file_path, silent=True):
    with open(file_path, "rb") as f:
        obj = joblib.load(f)
    if not silent:
        log.info(f"=> loaded {file_path}")
    return obj


def save_pickle(obj, file_path, silent=True):
    with open(file_path, "wb") as f:
        joblib.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    if not silent:
        log.info(f"=> saved to {file_path}")
