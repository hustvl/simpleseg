import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleAttention(nn.Module):

    def __init__(self, in_channels, inter_channels, out_channels, num_points, num_heads=1):
        super().__init__()
        self.num_points = num_points
        self.num_heads = num_heads
        self.offset = nn.Conv2d(in_channels, num_points * 2, 3, padding=1)
        self.q = nn.Conv2d(in_channels, inter_channels, 1, padding=0)
        self.k = nn.Linear(in_channels, inter_channels)
        self.v = nn.Linear(in_channels, in_channels)
        self.proj = nn.Linear(in_channels, out_channels)
        self.scale = (in_channels // num_heads) ** -0.5
        # init offset
        nn.init.constant_(self.offset.weight, 0.)
        nn.init.uniform_(self.offset.bias, -1.0, 1.0)

    @torch.no_grad()
    def compute_locations(self, query):
        h, w = query.size(2), query.size(3)
        y_loc = torch.linspace(-1, 1, h, device=query.device)
        x_loc = torch.linspace(-1, 1, w, device=query.device)
        locations = torch.stack(torch.meshgrid(y_loc, x_loc)).view(2, h * w).transpose(0, 1)
        return locations.to(query).requires_grad_(False)

    def forward(self, query, features):
        ref_points = self.compute_locations(query)
        B = query.size(0)
        H, W = query.shape[2:]
        size = torch.tensor([H, W]).to(query)
        # print(size)
        num_points = self.num_points
        offsets = self.offset(query).view(B, num_points, 2, H * W).transpose(2, 3) / size
        # (-1,1), BxNpx(HW)x2
        absolute_coords = (offsets + ref_points).reshape(B, num_points * H * W, 1, 2)
        # B, C H*W*N
        points = F.grid_sample(features, absolute_coords, mode='bilinear', align_corners=False)
        # -> B, H*W, num_points, C -> BHW, N, C
        points = points.reshape(B, -1, num_points, H*W).permute(0, 3, 2, 1).reshape(B*H*W, num_points, -1)

        q = self.q(query).permute(0, 2, 3, 1).reshape(B*H*W, self.num_heads, 1, -1) # BxCxHxW -> (BHW)xnhx1xC
        k = self.k(points).reshape(B*H*W, num_points, self.num_heads, -1).permute(0,2,3,1)
        v = self.v(points).reshape(B*H*W, num_points, self.num_heads, -1).permute(0,2,1,3)
        # add position encoding, (BHW)xnhx1xnum_points
        attn = F.softmax((q @ k) * self.scale, dim=-1)
        out = (attn @ v).reshape(B, H, W, -1)
        out = self.proj(out).permute(0, 3, 1, 2).contiguous()
        return out
