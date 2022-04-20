from data import build_data_module
from model import VQABaseModule
import argparse
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import (
    RichModelSummary,
    RichProgressBar,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    GPUStatsMonitor,
)
from pytorch_lightning.tuner.lr_finder import lr_find
import hydra
from pytorch_lightning.loggers import WandbLogger
import warnings
from util.log_utils import get_logger
import wandb
import os
from omegaconf import DictConfig

log = get_logger(__name__)


def build_callbacks(conf):
    callbacks = [RichModelSummary(max_depth=-1)]
    # callbacks += [LearningRateMonitor(logging_interval="epoch")]
    callbacks += [RichProgressBar()]
    ckpt_dirpath = "./"
    callbacks.append(hydra.utils.instantiate(conf.callbacks.early_stop))
    callbacks.append(
        hydra.utils.instantiate(conf.callbacks.checkpoint, dirpath=ckpt_dirpath)
    )
    return callbacks


@hydra.main(config_path="configs", config_name="train")
def main(conf: DictConfig):
    seed_everything(conf.seed)
    """instantiate running param"""
    conf.data.output_dir = os.getcwd()
    conf.data.root_dir = conf.data.root_dir
    conf.data.data_dir = conf.data.data_dir
    conf.data.cache_dir = conf.data.cache_dir
    if conf.ignore_warning:
        warnings.filterwarnings("ignore")

    if conf.debug:
        log.info("Debug mode activated!")
        conf.logger.mode = "disabled"
        conf.train.prefetch_factor = 2
        conf.train.num_workers = 0
        conf.gpus = 1

    data = build_data_module(conf)
    pre_emb = data.cache_dict["pre_emb"]
    model = VQABaseModule(conf, pre_emb)

    logger = WandbLogger(
        project=conf.logger.project,
        mode=conf.logger.mode,
        tags=conf.logger.tags,
        notes=conf.logger.notes,
    )
    if conf.train.watch_model:
        log.info("model watched")
        logger.watch(model, log_freq=conf.train.val_interval)

    callbacks = build_callbacks(conf)

    trainer = pl.Trainer(
        max_epochs=conf.train.num_epoch,
        check_val_every_n_epoch=conf.train.val_interval,
        gradient_clip_val=conf.train.grad_clip,
        gpus=conf.gpus,
        callbacks=callbacks,
        logger=logger,
        benchmark=True,
        strategy=conf.train.strategy,
        fast_dev_run=conf.train.fast_dev_run,
        accumulate_grad_batches=conf.train.accum_batch,
        precision=conf.train.precision,
    )

    trainer.fit(model, data)

    log.info("wandb finish")
    wandb.finish()


if __name__ == "__main__":
    main()
