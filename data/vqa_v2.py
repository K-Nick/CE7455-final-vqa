import sys

sys.path.insert(0, "..")
import os
from util.io import load_pickle, save_pickle, load_json
from .vocab import build_vocab, load_pretrained_emb
import subprocess
from util.log_utils import get_logger
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from tqdm.contrib.concurrent import thread_map
from .base_datamodule import VQABaseDataModule
from .dataset import VQABaseDataset, CollateFuncBuilder

log = get_logger(__name__)


class VQAv2DataModule(VQABaseDataModule):
    def __init__(self, conf):
        super().__init__(conf)

    def _download_data(self):
        data_dir = self.conf.data.data_dir
        dataset_name = self.conf.data.dataset_name
        dataset_dir = os.path.join(data_dir, dataset_name)
        flag_file = os.path.join(dataset_dir, f".download.flag")

        if os.path.exists(flag_file):
            # cached
            return
        url_list = [
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
            "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
            "http://images.cocodataset.org/zips/train2014.zip",
            "http://images.cocodataset.org/zips/val2014.zip",
        ]
        os.makedirs(dataset_dir, exist_ok=True)
        for url in url_list:
            os.makedirs(dataset_dir, exist_ok=True)
            filename = url.split("/")[-1]
            subprocess.run(
                f"cd {dataset_dir} && curl {url} -o {filename}&& unzip {filename} && rm -rf {filename}",
                shell=True,
            )

        subprocess.run(f"touch {flag_file}", shell=True)

    def _prepare_data(self, scope, conf):
        data_dir = conf.data.data_dir
        dataset_dir = os.path.join(data_dir, conf.data.dataset_name)

        self._download_data()
        annotations = load_json(
            os.path.join(dataset_dir, f"v2_mscoco_{scope}2014_annotations.json")
        )
        questions = load_json(
            os.path.join(dataset_dir, f"v2_OpenEnded_mscoco_{scope}2014_questions.json")
        )

        questions = questions["questions"]
        annotations = annotations["annotations"]

        def convert_single_zip(it):
            question, annotation = it
            sentence = question["question"]
            sentence = sentence.lower()
            sentence = sentence.replace(",", "").replace("?", "").replace("'s", " 's")
            ret_question = sentence.split()
            ret_annotation = [a["answer"] for a in annotation["answers"]]
            gt_annot = annotation["multiple_choice_answer"]
            image_id = question["image_id"]
            image_filename = f"{image_id}.pkl"
            image_path = os.path.join(data_dir, f"{scope}_img_feat", image_filename)

            meta_dict = {
                "q_id": question["question_id"],
                "q_str": question["question"],
                "a_str": "|".join(ret_annotation),
                "img_id": image_id,
                "q_type": annotation["answer_type"],
            }

            return ret_question, ret_annotation, gt_annot, image_path, meta_dict

        questions, annotations, gt_annots, image_paths, meta_dict = zip(
            *thread_map(
                convert_single_zip,
                list(zip(questions, annotations)),
                chunksize=1024,
                max_workers=256,
                desc=f"prepare {scope} data...",
            )
        )

        # unzipped form will be like ((item0,), (item1, )....)
        questions = list(questions)
        annotations = list(annotations)
        image_paths = list(image_paths)
        meta_dict = list(meta_dict)
        gt_annots = list(gt_annots)

        return questions, annotations, gt_annots, image_paths, meta_dict

    # return train_dataset, val_dataset, train_loader, val_loader, cache_dict
    def _build_loader(self):
        conf = self.conf

        root_dir = conf.data.data_dir
        dataset_dir = os.path.join(root_dir, conf.data.dataset_name)

        if conf.data.min_ans_freq:
            cache_path = os.path.join(
                conf.data.cache_dir, f"data_cache.freq{conf.data.min_ans_freq}.pkl"
            )
        else:
            cache_path = os.path.join(
                conf.data.cache_dir, f"data_cache.max_tok{conf.data.max_ans_tokens}.pkl"
            )

        if conf.data.use_cache and os.path.exists(cache_path):
            cache_dict = load_pickle(cache_path)
            train_dataset = cache_dict["train_dataset"]
            val_dataset = cache_dict["val_dataset"]
            q_vocab = cache_dict["q_vocab"]
            a_vocab = cache_dict["a_vocab"]
            pre_emb = cache_dict["pre_emb"]

            log.info("=> datasets & vocab cache loaded")
        else:
            (
                train_questions,
                train_annotations,
                train_gt_annot,
                train_image_paths,
                train_meta_dict,
            ) = self._prepare_data("train", conf)
            (
                val_questions,
                val_annotations,
                val_gt_annots,
                val_image_paths,
                val_meta_dict,
            ) = self._prepare_data("val", conf)
            q_vocab, a_vocab = build_vocab(
                conf, train_questions + val_questions, train_gt_annot + val_gt_annots
            )
            pre_emb = load_pretrained_emb(conf, q_vocab, conf.data.word_dim)

            train_dataset = VQABaseDataset(
                conf,
                train_questions,
                train_annotations,
                train_image_paths,
                train_meta_dict,
                q_vocab,
                a_vocab,
            )
            val_dataset = VQABaseDataset(
                conf,
                val_questions,
                val_annotations,
                val_image_paths,
                val_meta_dict,
                q_vocab,
                a_vocab,
            )

            cache_dict = {
                "train_dataset": train_dataset,
                "val_dataset": val_dataset,
                "q_vocab": q_vocab,
                "a_vocab": a_vocab,
                "pre_emb": pre_emb,
            }

            if conf.data.use_cache:
                save_pickle(cache_dict, cache_path, silent=False)

        # save num_ans in config, <unk> should be ignored
        conf.data.num_ans = len(a_vocab.get_itos())

        train_loader = DataLoader(
            train_dataset,
            batch_size=conf.train.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=CollateFuncBuilder(conf, is_train=True),
            prefetch_factor=conf.train.prefetch_factor,
            num_workers=conf.train.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=conf.train.batch_size,
            pin_memory=True,
            collate_fn=CollateFuncBuilder(conf, is_train=False),
            prefetch_factor=conf.train.prefetch_factor,
            num_workers=conf.train.num_workers,
        )

        return train_dataset, val_dataset, train_loader, val_loader, cache_dict
