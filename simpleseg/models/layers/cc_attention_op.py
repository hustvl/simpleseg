import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import Softmax


def NEG_INF_DIAG(n, device):
    return torch.diag(torch.tensor(float("-inf")).to(device).repeat(n), 0)


class CrissCrossAttentionOps(nn.Module):
    """Criss-Cross Attention Module"""

    def __init__(self, in_dim, inter_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=inter_dim, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=inter_dim, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.softmax = Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, query, key, value, return_weight=False):  # 使用einops
        m_batchsize, _, height, width = query.shape
        proj_query = self.query_conv(query)
        proj_query_H = rearrange(proj_query, "B C H W -> (B W) H C")
        proj_query_W = rearrange(proj_query, "B C H W -> (B H) W C")
        proj_key = self.key_conv(key)
        proj_key_H = rearrange(proj_key, "B C H W -> (B W) C H")
        proj_key_W = rearrange(proj_key, "B C H W -> (B H) C W")
        proj_value = self.value_conv(value)
        proj_value_H = rearrange(proj_value, "B C H W -> (B W) C H")
        proj_value_W = rearrange(proj_value, "B C H W -> (B H) C W")
        energy_H = rearrange(
            (torch.bmm(proj_query_H, proj_key_H) +
             NEG_INF_DIAG(height, proj_query_H.device)),
            "(B W) H H2 -> B H W H2",
            B=m_batchsize,
        )
        energy_W = rearrange(
            torch.bmm(proj_query_W, proj_key_W), "(B H) W W2 -> B H W W2", B=m_batchsize
        )
        # energy_H = torch.einsum('bchw,bcHw->bhwH',proj_query,proj_key)
        # energy_W = torch.einsum('bchw,bchW->bhwW',proj_query,proj_key)
        concate = self.softmax(
            torch.cat([energy_H, energy_W], -1))  # B,H,W,(H+W)
        att_H = rearrange(concate[:, :, :, :height], "B H W H2 -> (B W) H2 H")
        att_W = rearrange(concate[:, :, :, height:], "B H W W2 -> (B H) W2 W")
        out = rearrange(
            torch.bmm(proj_value_H, att_H), "(B W) C H -> B C H W", B=m_batchsize
        )
        out += rearrange(
            torch.bmm(proj_value_W, att_W), "(B H) C W -> B C H W", B=m_batchsize
        )
        out *= self.gamma
        out += value
        if return_weight:
            return out, concate
        return out
