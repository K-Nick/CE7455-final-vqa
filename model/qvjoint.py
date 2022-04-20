import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import PositionalEncoding


class SelfAttnBlock(nn.Module):
    def __init__(self, d_model, num_head, ff_dim, num_layer, dropout) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_head, ff_dim, num_layer, dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)

    def forward(self, embs):
        return self.encoder(embs)


class QVJointBaseModel(nn.Module):
    def __init__(self, conf, pre_emb) -> None:
        super().__init__()
        word_dim = conf.data.word_dim
        img_dim = conf.data.img_dim
        dropout = conf.model.dropout
        num_ans = conf.data.num_ans
        num_hid = conf.model.num_hid

        q_enc_conf = conf.model.q_enc

        self.pe = PositionalEncoding(
            d_model=q_enc_conf.d_model, dropout=conf.model.dropout
        )
        self.q_enc_adapt = nn.Linear(word_dim, q_enc_conf.d_model)

    def forward(self, v_emb, b, qs, q_lens):
        q_emb = self.emb(qs)
        q_emb = self.q_enc_adapt(q_emb)
        q_emb = self.q_enc(q_emb, q_lens)
        0


        q_emb = self.q_proj(q_emb)
        v_emb = self.v_proj(v_emb)

        if conf.model.qv_attn.cls_fuse:
            self.cls_emb = self.cls_emb.type_as(v_emb)
            qv_attn = self.qv_attn(q_emb, v_emb, q_lens, self.cls_emb)
        else:
            qv_attn = self.qv_attn(q_emb, v_emb, q_lens)

        logit = self.clf(qv_attn)
        return logit
