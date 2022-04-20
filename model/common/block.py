import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


class FCBlock(nn.Module):
    def __init__(self, dims, activation=nn.LeakyReLU, dropout=0.0, logit=False) -> None:
        """
        fully connected block with weight initialization
        dims[0] = input
        dims[i] i>0 means output dimension of respective layers
        """
        super().__init__()

        layers = []

        in_dim = dims[0]
        num_layer = len(dims) - 1
        for idx, out_dim in enumerate(dims[1:]):
            layers += [nn.Linear(in_dim, out_dim)]
            in_dim = out_dim
            if idx != num_layer - 1:
                layers += [
                    activation()
                ]  # ignore last activation in case of logit output

        self.fc_net = nn.Sequential(*layers)
        self.out_activation = activation()
        self.logit = logit

    def forward(self, x):
        out = self.fc_net(x)

        if not self.logit:
            out = self.out_activation(out)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        pe = self.pe[: x.shape[1]].transpose(0, 1)
        x = x + pe
        return self.dropout(x)


class MaskedRNN(nn.Module):
    def __init__(
        self,
        rnn,
        input_size,
        hidden_size,
        bidirectional=True,
        num_layers=1,
        dropout=0.0,
    ) -> None:
        super().__init__()
        rnn_cls = eval("nn." + rnn)
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, src, src_lens):
        """return the final state of RNN"""
        outputs = self.forward_all(src, src_lens)

        if self.rnn.bidirectional:
            output_hidden = outputs[:, -1, : self.num_hid + 1]
            backward_emb = outputs[:, 0, self.num_hid + 1 :]
            output_hidden = torch.cat([output_hidden, backward_emb], dim=1)
        else:
            output_hidden = outputs[:, -1, :]

        return output_hidden

    def forward_all(self, src, src_lens):
        """return the output of RNN directly"""
        src = pack_padded_sequence(
            src, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.rnn(src)
        outputs = pad_packed_sequence(outputs, batch_first=True)

        return outputs
