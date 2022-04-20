import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import MaskedRNN, PositionalEncoding
from util.nn_utils import get_mask_from_lens
import numpy as np


class MLPNet(nn.Module):
    def __init__(
        self, input_size, num_hids: list, activation="relu", logit=False
    ) -> None:
        super().__init__()
        input_size = input_size
        layer_list = []
        nlayer = len(num_hids)
        self.act = activation
        for idx, num_hid in enumerate(num_hids):
            layer_list.append(nn.Linear(input_size, num_hid))
            input_size = num_hid
            if idx != nlayer - 1 or not logit:
                layer_list += [self.get_activation()]

        self.net = nn.Sequential(*layer_list)

    def get_activation(self):
        name = self.act
        if name == "relu":
            return nn.ReLU()

    def forward(self, x):
        return self.net(x)


class QVGuideAttn(nn.Module):
    def __init__(self, d_model, nhead, ff_dim, nlayer, dropout=0.0) -> None:
        """Implement MCAN-ED Guided Attention"""
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            ff_dim,
            layer_norm_eps=1.0e-6,
            batch_first=True,
            dropout=dropout,
        )
        self.q_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)

        guided_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            ff_dim,
            layer_norm_eps=1.0e-6,
            batch_first=True,
            dropout=dropout,
        )
        self.qv_guide_decoder = nn.TransformerDecoder(guided_layer, num_layers=nlayer)

    def forward(self, q_emb, v_emb, q_lens):
        q_len_mask = get_mask_from_lens(q_lens, max_len=q_emb.shape[1])
        q_emb = self.q_encoder(q_emb, src_key_padding_mask=~q_len_mask)
        v_emb = self.qv_guide_decoder(v_emb, q_emb, memory_key_padding_mask=~q_len_mask)

        return q_emb, v_emb


class AttnReducer(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.attn_func = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x, lens=None):
        """
        x (bsz, len, dim) -> (bsz, dim)
        """
        mask = None
        if lens is not None:
            mask = ~get_mask_from_lens(lens, x.shape[1])

        return self.attn(x, mask)

    def attn(self, x, mask):
        attn_score: torch.Tensor = self.attn_func(x).squeeze(2)  # (bsz, len, 1)
        if mask is not None:
            attn_score.masked_fill_(mask, value=-np.inf)
        attn_score = F.softmax(attn_score, dim=1)
        x = torch.bmm(x.transpose(1, 2), attn_score.unsqueeze(2)).squeeze(
            2
        )  # (bsz, dim)

        return x


class QVGuideModel(nn.Module):
    def __init__(self, conf, pre_emb) -> None:
        super().__init__()
        word_dim = conf.data.word_dim
        img_dim = conf.data.img_dim
        num_ans = conf.data.num_ans - 1
        num_hid = conf.model.num_hid
        clf_ff_dim = conf.model.clf_ff_dim
        qv_cross_conf = conf.model.qv_cross

        self.emb = nn.Embedding.from_pretrained(pre_emb, padding_idx=0)
        self.pe = PositionalEncoding(qv_cross_conf.d_model)
        self.gru = MaskedRNN("GRU", input_size=word_dim, hidden_size=num_hid)

        self.q_proj = MLPNet(num_hid * 2, [num_hid])
        self.v_proj = MLPNet(img_dim, [num_hid])

        self.qv_cross = QVGuideAttn(
            d_model=qv_cross_conf.d_model,
            nhead=qv_cross_conf.nhead,
            ff_dim=qv_cross_conf.ff_dim,
            nlayer=qv_cross_conf.nlayer,
            dropout=0.0,
        )

        self.q_reducer = AttnReducer(num_hid, num_hid)
        self.v_reducer = AttnReducer(num_hid, num_hid)

        self.norm = nn.LayerNorm(qv_cross_conf.d_model, eps=1e-6)

        self.clf = nn.Sequential(
            nn.Linear(num_hid * 2, clf_ff_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(clf_ff_dim, num_ans),
        )

    def forward(self, img_feats, img_qs_feats, qs, q_lens):
        q_emb = self.emb(qs)
        q_emb, _ = self.gru.forward_all(q_emb, q_lens)

        q_emb = self.pe(q_emb)

        q_emb = self.q_proj(q_emb)  # (bsz, seq_len, word_dim -> num_hid)
        v_emb = self.v_proj(img_feats)  # (bsz, bbox, img_dim -> num_hid)

        q_emb, v_emb = self.qv_cross(q_emb, v_emb, q_lens)

        q_emb = self.q_reducer(q_emb, q_lens)
        v_emb = self.v_reducer(v_emb)

        qv_emb = torch.cat([q_emb, v_emb], dim=1)
        qv_emb = self.norm(qv_emb)
        logit = self.clf(qv_emb)

        return logit

    # class QVGuideModel(nn.Module):
    #     def __init__(self, conf, pre_emb) -> None:
    #         super().__init__()
    #         word_dim = conf.data.word_dim
    #         img_dim = conf.data.img_dim
    #         num_ans = conf.data.num_ans - 1
    #         num_hid = 1024
    #         dropout = 0.5

    #         self.emb = nn.Embedding.from_pretrained(pre_emb, padding_idx=0)
    #         self.gru = MaskedRNN("GRU", input_size=word_dim, hidden_size=num_hid)

    #         self.q_proj = MLPNet(num_hid * 2, [num_hid])
    #         self.v_proj = MLPNet(img_dim, [num_hid])

    #         self.qv_cross = QVGuideAttn(d_model=num_hid, nhead=8, ff_dim=2048, nlayer=3)

    #         self.q_reducer = AttnReducer(num_hid, num_hid)
    #         self.v_reducer = AttnReducer(num_hid, num_hid)

    #         self.clf = nn.Sequential(
    #             nn.Linear(num_hid * 2, num_hid),
    #             nn.ReLU(),
    #             nn.Dropout(dropout),
    #             nn.Linear(num_hid, num_ans),
    #         )

    def forward(self, img_feats, img_qs_feats, qs, q_lens):
        q_emb = self.emb(qs)
        q_emb, _ = self.gru.forward_all(q_emb, q_lens)

        q_emb = self.q_proj(q_emb)  # (bsz, seq_len, word_dim -> num_hid)
        v_emb = self.v_proj(img_feats)  # (bsz, bbox, img_dim -> num_hid)

        q_emb, v_emb = self.qv_cross(q_emb, v_emb, q_lens)

        q_emb = self.q_reducer(q_emb, q_lens)
        v_emb = self.v_reducer(v_emb)

        qv_emb = torch.cat([q_emb, v_emb], dim=1)
        logit = self.clf(qv_emb)

        return logit
