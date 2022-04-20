import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import MaskedRNN
from util.nn_utils import get_mask_from_lens
import numpy as np
from .qvguide import MLPNet, AttnReducer


class QVCrossAttn(nn.Module):
    def __init__(self, d_model, nhead, ff_dim, nlayer, dropout) -> None:
        super().__init__()

        vgq_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, ff_dim, layer_norm_eps=1.e-6, batch_first=True, dropout=dropout)
        self.vgq_decoder = nn.TransformerDecoder(vgq_decoder_layer, num_layers=nlayer)

        qgv_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, ff_dim, layer_norm_eps=1.e-6, batch_first=True, dropout=dropout)
        self.qgv_decoder = nn.TransformerDecoder(qgv_decoder_layer, num_layers=nlayer)
    
    def forward(self, q_emb, v_emb, q_lens):
        q_len_mask = get_mask_from_lens(q_lens, max_len=q_emb.shape[1])
        q_emb = self.vgq_decoder(q_emb, v_emb, tgt_key_padding_mask=~q_len_mask)
        v_emb = self.qgv_decoder(v_emb, q_emb, memory_key_padding_mask=~q_len_mask)

        return q_emb, v_emb



class QVCrossModel(nn.Module):
    def __init__(self, conf, pre_emb) -> None:
        super().__init__()
        word_dim = conf.data.word_dim
        img_dim = conf.data.img_dim
        num_ans = conf.data.num_ans - 1
        num_hid = conf.model.num_hid
        clf_ff_dim = conf.model.clf_ff_dim
        dropout = 0.5
        qv_cross_conf = conf.model.qv_cross
        
        self.emb = nn.Embedding.from_pretrained(pre_emb, padding_idx=0)
        self.gru = MaskedRNN("GRU", input_size=word_dim, hidden_size=num_hid)

        self.q_proj = MLPNet(num_hid * 2, [num_hid])
        self.v_proj = MLPNet(img_dim, [num_hid])

        self.qv_cross = QVCrossAttn(d_model=qv_cross_conf.d_model,
                                    nhead=qv_cross_conf.nhead,
                                    ff_dim=qv_cross_conf.ff_dim,
                                    nlayer=qv_cross_conf.nlayer,
                                    dropout=0.3)
        
        self.q_reducer = AttnReducer(num_hid, num_hid)
        self.v_reducer = AttnReducer(num_hid, num_hid)
        
        self.clf = nn.Sequential(nn.Linear(num_hid * 2, clf_ff_dim),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(clf_ff_dim, num_ans))


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
        