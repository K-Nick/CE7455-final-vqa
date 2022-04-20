import sys

sys.path.insert(0, "..")
from util.io import load_pickle
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from data.vocab import AnswerNormalizer


class CollateFuncBuilder:
    def __init__(self, conf, is_train) -> None:
        self.num_ans = conf.data.num_ans
        self.is_train = is_train
        self.conf = conf

    def __call__(self, data):
        batch_size = len(data)
        conf = self.conf

        image_path, questions, annotations, meta_dict = zip(*data)
        image_feats_dict = [load_pickle(fp) for fp in image_path]

        img_feats = torch.from_numpy(
            np.concatenate([it["img_features"][None, :] for it in image_feats_dict])
        )
        img_spatial_feats = torch.from_numpy(
            np.concatenate([it["spatial_features"][None, :] for it in image_feats_dict])
        )

        target_list = []
        for group_annot in annotations:
            target = torch.zeros(1, self.num_ans)  # compute answer frequency
            for annot in group_annot:
                target[0, annot] += 1
            target_list.append(target)

        target = torch.cat(target_list, dim=0)
        target = target[:, 1:]  # cut off <unk> which is idx 0

        target = torch.clip_(target * 0.3, max=1)  # validation target

        # deal with variable length
        q_lens = torch.tensor([q.shape[0] for q in questions])
        questions = [torch.from_numpy(q) for q in questions]
        if conf.data.pre_pad:
            questions = [torch.flip(q, dims=[0]) for q in questions]
            questions = pad_sequence(questions, batch_first=True)
            questions = torch.flip(questions, dims=[1])
        else:
            questions = pad_sequence(questions, batch_first=True)

        return meta_dict, img_feats, img_spatial_feats, questions, q_lens, target


class VQABaseDataset(Dataset):
    def __init__(
        self, conf, questions, annotations, image_paths, meta_dict, q_vocab, a_vocab
    ):
        super().__init__()

        # convert tokens to indices
        questions = [np.array(q_vocab.lookup_indices(q)) for q in questions]

        ans_norm = AnswerNormalizer()
        annotations = [
            [ans_norm.preprocess_answer(a) for a in annot_group]
            for annot_group in annotations
        ]
        annotations = [
            a_vocab.lookup_indices(annot_group) for annot_group in annotations
        ]
        # keep <unk> and ignore them in collate_fn

        self.questions = questions
        self.annotations = annotations
        self.image_paths = image_paths
        self.meta_dict = meta_dict

    def __getitem__(self, idx):
        question = self.questions[idx]
        annotation = self.annotations[idx]
        image_path = self.image_paths[idx]
        meta_dict = self.meta_dict[idx]
        meta_dict["idx"] = idx

        return image_path, question, annotation, meta_dict

    def __len__(self):
        return len(self.questions)
