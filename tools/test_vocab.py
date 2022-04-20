import sys

sys.path.insert(0, "..")
from util.io import load_pickle

if __name__ == "__main__":
    cache_dict = load_pickle("../cache_bin/data_cache.freq9.pkl", silent=False)
    a_vocab = cache_dict["a_vocab"]
    q_vocab = cache_dict["q_vocab"]
    val_set = cache_dict["val_dataset"]
    import ipdb

    ipdb.set_trace()  # FIXME
