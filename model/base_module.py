import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pytorch_lightning import LightningModule
import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as opt
from torch.optim.lr_scheduler import CosineAnnealingLR, ChainedScheduler, ExponentialLR
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from util.log_utils import get_logger
from omegaconf import DictConfig
from baseline.base_model import build_baseline0_newatt_interface
from .peterson_base import PetersonBaseline
from .qvhadm import QVHadamard
from .qvjoint import QVJointBaseModel
from .qvguide import QVGuideModel
from .qvcross import QVCrossModel

log = get_logger(__name__)


def build_model(conf, pre_emb):
    if conf.model.arch == "qvjoint":
        return QVJointBaseModel(conf, pre_emb)

    elif conf.model.arch == "peterson_base":
        return PetersonBaseline(conf, pre_emb)

    elif conf.model.arch == "qvhadm":
        return QVHadamard(conf, pre_emb)

    elif conf.model.arch == "qvguide":
        return QVGuideModel(conf, pre_emb)

    elif conf.model.arch == "qvcross":
        return QVCrossModel(conf, pre_emb)


class VQABaseModule(LightningModule):
    def __init__(self, conf: DictConfig, pre_emb):
        super().__init__()
        self.conf = conf.copy()

        self.net = build_model(conf, pre_emb)
        self.save_hyperparameters()

    def setup(self, stage):
        if self.logger:
            self.logger.experiment.define_metric("val/acc", summary="max")
            self.logger.experiment.log_code(self.conf.data.root_dir)

    def forward(self, v_emb, b, qs, q_lens):
        return self.net(v_emb, b, qs, q_lens)

    def training_step(self, batch, batch_idx):
        meta_dict, img_feats, img_spatial_feats, questions, q_lens, target = batch
        logit = self.net(img_feats, img_spatial_feats, questions, q_lens)
        # loss = F.cross_entropy(logit, target)
        prob = F.sigmoid(logit)
        loss = F.binary_cross_entropy(prob, target)
        return {"loss": loss}

    def training_epoch_end(self, outputs) -> None:
        epoch_loss = torch.mean(torch.stack([it["loss"] for it in outputs])).item()

        log.info(f"train/ce_loss: {epoch_loss:.4f}")
        self.log_dict(
            {
                "train/ce_loss": epoch_loss,
            },
            on_step=False,
            on_epoch=True,
        )
        if self.conf.optim.scheduler:
            self.log_dict(
                {
                    "train/lr": self.trainer.lr_schedulers[0][
                        "scheduler"
                    ].get_last_lr()[0]
                },
                on_step=False,
                on_epoch=True,
            )

    def validation_step(self, batch, batch_idx):
        meta_dict, img_feats, img_spatial_feats, questions, q_lens, target = batch
        logit = self.net(img_feats, img_spatial_feats, questions, q_lens)
        pred_arr = torch.argmax(logit, dim=1)  # (B, N_ans)
        pred_arr = F.one_hot(pred_arr, num_classes=self.conf.data.num_ans - 1)

        acc = torch.sum(pred_arr * target, dim=1)

        return {"acc": acc, "logit": logit, "meta_dict": meta_dict}

    def validation_epoch_end(self, outputs):
        cat_output = torch.cat([o["acc"] for o in outputs])
        acc = torch.mean(cat_output)
        log.info(f"val/acc: {acc:.4f}")
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        AdamW + Chained Scheduler (CosineAnnealing + Exponential)
        """
        conf = self.conf
        optim_conf = conf.optim
        optimizer_list = []
        scheduler_list = []

        optimizer = hydra.utils.instantiate(
            optim_conf.optimizer, params=self.net.parameters()
        )

        optimizer_target = optim_conf.optimizer.target
        if optimizer_target == "adamw":
            optimizer = opt.AdamW(self.net.parameters(), lr=optim_conf.optimizer.lr)
        elif optimizer_target == "adamax":
            optimizer = opt.Adamax(self.net.parameters(), lr=optim_conf.optimizer.lr)
        elif optimizer_target == "adadelta":
            optimizer = opt.Adadelta(self.net.parameters(), lr=optim_conf.optimizer.lr)
        optimizer_list.append(optimizer)

        scheduler_dict = None
        if optim_conf.scheduler:
            scheduler_target = optim_conf.scheduler.target
            lr_scheduler = None

            # setup scheduler
            if scheduler_target == "cosine":
                scheduler_conf = optim_conf.scheduler.cosine
                lr_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_conf.T_max,
                    eta_min=optim_conf.opitmizer.lr / 50,
                )

            elif scheduler_target == "exp":
                scheduler_conf = optim_conf.scheduler.exp
                lr_scheduler = ExponentialLR(
                    optimizer=optimizer, gamma=scheduler_conf.gamma
                )

            elif scheduler_target == "warmup_cosine":
                scheduler_conf = optim_conf.scheduler.warmup_cosine
                lr_scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer=optimizer,
                    warmup_epochs=scheduler_conf.warmup_epochs,
                    max_epochs=scheduler_conf.T_max,
                    warmup_start_lr=optim_conf.optimizer.lr / 50,
                    eta_min=optim_conf.optimizer.lr / 50,
                )
            # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ChainedScheduler.html#torch.optim.lr_scheduler.ChainedScheduler
            # lr_scheduler = ChainedScheduler([cos_scheduler, exp_scheduler])

            scheduler_dict = {"scheduler": lr_scheduler, "interval": "epoch"}
            scheduler_list.append(scheduler_dict)

        return optimizer_list, scheduler_list
