import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn
from model.common.block import FCBlock
import torch
import torch.nn.functional as F


class QVBilinearAttention(nn.Module):
    """
    implementation of bottom up attention from https://arxiv.org/abs/1707.07998
    return attention score
    """

    def __init__(self, hid_nums, q_dim, v_dim) -> None:
        super().__init__()
        self.fc_net = FCBlock([q_dim + v_dim] + list(hid_nums))
        self.linear = nn.Linear(hid_nums[-1], 1)

    def forward(self, v_emb, q_emb):
        """
        v: [batch, num_bbox, v_dim]
        q: [batch, q_dim]
        return: v_attn
        """
        num_boxx = v_emb.shape[1]
        q_emb = q_emb.unsqueeze(1).repeat((1, num_boxx, 1))
        qv_concat = torch.cat([q_emb, v_emb], dim=2)
        feat = self.fc_net(qv_concat)  # [batch, num_bbox, hid_num]
        attn_score = self.linear(feat).squeeze(2)
        attn_score = F.softmax(attn_score, dim=1)

        return attn_score


class QVMultAttention(nn.Module):
    def __init__(self, q_dim, v_dim, num_hid) -> None:
        super().__init__()
        self.q_proj = FCBlock([q_dim, num_hid], residual=False)
        self.v_proj = FCBlock([v_dim, num_hid], residual=False)
        self.linear = nn.Linear(num_hid, 1)

    def forward(self, v_emb, q_emb):
        # mask = get_mask_from_lens(q_lens)
        num_boxx = v_emb.shape[1]
        q_emb = q_emb.unsqueeze(1).repeat((1, num_boxx, 1))
        q_emb = self.q_proj(q_emb)
        # q_emb = torch.einsum("ijk, ij->ijk", q_emb, mask.int().float())
        v_emb = self.v_proj(v_emb)
        qv_emb = q_emb * v_emb
        logit = self.linear(qv_emb).squeeze(2)
        attn_score = F.softmax(logit, dim=1)

        return attn_score
