import sys

sys.path.insert(0, "..")
import pytorch_lightning as pl
import subprocess
import os
from .vqa_v2 import VQAv2DataModule
import torch
from util.io import load_pickle
import numpy as np
import torch


def build_data_module(conf):
    return VQAv2DataModule(conf)


class VQABaseDataModule(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        train_set, val_set, train_loader, val_loader, cache_dict = self._build_loader(
            conf
        )

        self.train_set = train_set
        self.val_set = val_set
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cache_dict = cache_dict

    def _prepare_and_unzip_files(url, root_dir):
        os.makedirs(root_dir, exist_ok=True)
        filename = url.split("/")[-1]
        subprocess.run(
            f"cd {root_dir} && curl {url} -o {filename}&& unzip {filename} && rm -rf {filename}",
            shell=True,
        )

    def _download_data(self):
        raise NotImplementedError()

    def _prepare_data(self):
        raise NotImplementedError()

    def _build_loader(self):
        raise NotImplementedError()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.val_loader
