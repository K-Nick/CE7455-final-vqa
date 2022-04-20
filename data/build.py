import sys

sys.path.insert(0, "..")
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as T
from torchtext.vocab import build_vocab_from_iterator, GloVe
from util.io import load_json, load_pickle, save_pickle
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map, thread_map
import torch.nn.functional as F
from util.log_utils import get_logger

log = get_logger(__name__)

# ============================================================================ #
# Utility Functions for extract data, loading pretrained embedding and building vocab


# ============================================================================ #
# VQA dataset


# ============================================================================ #
# build dataloader and collate_fn


# ============================================================================ #
# ! TEST
