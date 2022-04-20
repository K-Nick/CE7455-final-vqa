import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedTanh(nn.Module):
    """Deprecated"""

    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = F.tanh(self.linear1(x)) * F.sigmoid(self.linear2(x))
        return x


class ReLUNet(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class PetersonBaseline(nn.Module):
    def __init__(self, conf, pre_emb=None) -> None:
        super().__init__()
        q_dim = 300
        v_dim = 2048
        num_hid = 1024
        fa_num_hid = 1280
        dropout = 0.5
        num_ans = conf.data.num_ans

        self.emb = nn.Embedding.from_pretrained(pre_emb, padding_idx=0)
        self.gru = nn.GRU(q_dim, num_hid, batch_first=True, bidirectional=True)

        self.attn_fn = nn.Sequential(
            ReLUNet(2 * num_hid + v_dim, fa_num_hid), nn.Linear(fa_num_hid, 1)
        )

        self.q_proj = ReLUNet(2 * num_hid, num_hid)
        self.v_proj = ReLUNet(v_dim, num_hid)

        self.clf = nn.Sequential(
            ReLUNet(num_hid, num_hid),
            nn.Dropout(dropout),
            nn.Linear(num_hid, num_ans - 1),
        )
        # self.do = nn.Dropout(p=dropout)
        self.conf = conf

    def forward(self, v_emb, img_spatial, qs, q_lens):
        """detailed model is in https://arxiv.org/pdf/1708.02711.pdf"""
        bsz = qs.shape[0]

        q_emb = self.emb(qs)

        # sentence embedding
        outputs, hn = self.gru(q_emb)
        q_emb = hn.transpose(0, 1).reshape((bsz, -1))

        # l2 norm
        # v_emb = v_emb / torch.sqrt(torch.sum(v_emb * v_emb, dim=2, keepdim=True))

        # attention
        attn_score = self.attn_score(v_emb, q_emb)
        v_attn_emb = torch.bmm(attn_score.unsqueeze(1), v_emb).squeeze(1)
        # v_attn_emb = torch.einsum("ijk,ij->ik", v_emb, attn_score)

        # projection
        q_emb = self.q_proj(q_emb)
        v_emb = self.v_proj(v_attn_emb)

        qv_emb = q_emb * v_emb

        # qv_emb = self.dropout(qv_emb)
        logit = self.clf(qv_emb)

        return logit

    def attn_score(self, v_emb, q_emb):
        num_bbox = v_emb.shape[1]
        q_emb = q_emb.unsqueeze(1).repeat((1, num_bbox, 1))
        qv_emb = torch.cat([v_emb, q_emb], dim=2)
        logit = self.attn_fn(qv_emb).squeeze(2)

        score = F.softmax(logit, dim=1)

        return score
