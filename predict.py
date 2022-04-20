import sys

sys.path.insert(0, "..")
import torch
import pytorch_lightning as pl
from model.base_module import VQABaseModule
from omegaconf import open_dict
from data import build_data_module
import pandas as pd
from collections import OrderedDict


class CaptureValidationOutput(pl.Callback):
    # https://github.com/PyTorchLightning/pytorch-lightning/discussions/11659
    def __init__(self) -> None:
        self.val_outs = []

    def reset(self):
        self.val_outs = []

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.val_outs.append(outputs)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        return self.val_outs


def predict(ckpt_list):
    # https://stackoverflow.com/questions/66295334/create-a-new-key-in-hydra-dictconfig-from-python-file

    capture_callback = CaptureValidationOutput()
    trainer = pl.Trainer(
        logger=False,
        accelerator="auto",
        benchmark=True,
        callbacks=[capture_callback],
    )

    def get_df(name, ckpt):
        model = VQABaseModule.load_from_checkpoint(ckpt)
        conf = model.conf
        with open_dict(conf):
            conf.data.dataset_name = "VQA-v2"
            conf.data.data_dir = "./data_bin"
            conf.data.cache_dir = "./cache_bin"
            conf.data.root_dir = "./"

        data = build_data_module(conf)
        a_vocab = data.cache_dict["a_vocab"]

        capture_callback.reset()
        trainer.validate(model, data)

        outputs = capture_callback.val_outs
        pred = (
            torch.cat([torch.argmax(x["logit"], dim=1) + 1 for x in outputs])
            .cpu()
            .numpy()
            .tolist()
        )
        acc = torch.cat([x["acc"] for x in outputs]).cpu().numpy().tolist()
        acc = [f"{x:.3f}" for x in acc]

        meta_dict = []
        for output in outputs:
            meta_dict += output["meta_dict"]
        df = pd.DataFrame(meta_dict)

        df[f"{name}/pred"] = a_vocab.lookup_tokens(pred)
        df[f"{name}/acc"] = acc
        return df

    ret_df = get_df(*ckpt_list[0])

    for nm, ckpt in ckpt_list[1:]:
        df = get_df(nm, ckpt)
        ret_df[f"{nm}/pred"] = df[f"{nm}/pred"]
        ret_df[f"{nm}/acc"] = df[f"{nm}/acc"].astype(float)

    ret_df.to_csv("./cache_bin/exp_compare.csv")


def groupby_df(ckpt_list):
    ret_df = pd.read_csv("./cache_bin/exp_compare.csv")
    agg_acc = {}
    for nm, ckpt in ckpt_list:
        agg_acc[f"{nm}/acc"] = "mean"
        ret_df[f"{nm}/acc"] = ret_df[f"{nm}/acc"].astype(float)

    group_df = ret_df.groupby("q_type").agg(agg_acc)

    print(group_df)


if __name__ == "__main__":
    # ckpt = (
    #     "./logs/experiments/runs/default/2022-04-09_04-16-10/epoch=56-step=24738.ckpt"
    # )
    ckpt_list = [
        (
            "qvguide-best",
            "./logs/experiments/runs/default/2022-04-19_20-04-38/epoch=50-step=22134.ckpt",
        ),
        (
            "peterson-baseline",
            "./logs/experiments/runs/default/2022-04-16_13-47-18/epoch=35-step=15624.ckpt",
        ),
    ]

    predict(ckpt_list)
    # groupby_df(ckpt_list)
