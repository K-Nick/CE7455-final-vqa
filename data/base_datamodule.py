import pytorch_lightning as pl
import subprocess
import os


class VQABaseDataModule(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        train_set, val_set, train_loader, val_loader, cache_dict = self._build_loader()

        self.train_set = train_set
        self.val_set = val_set
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cache_dict = cache_dict

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
