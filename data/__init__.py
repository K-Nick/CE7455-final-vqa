from .vqa_v2 import VQAv2DataModule


def build_data_module(conf):
    return VQAv2DataModule(conf)
