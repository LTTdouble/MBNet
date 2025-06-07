
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y



class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dropout=0.0,
        conv_bias=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # Projection layers
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # State space parameters
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            )
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output layers
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        B, H, W, C = x.shape
        L = H * W
        K = 4  # Number of scanning directions
        dim = self.d_inner * K  # dim used for scanning

        # Project input
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (B, H, W, d_inner)

        # Conv activation
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, d_inner, H, W)
        x = self.act(self.conv2d(x))  # (B, d_inner, H, W)

        # Prepare scanning directions
        x_hwwh = torch.stack(
            [x.flatten(2), x.transpose(2, 3).flatten(2)],
            dim=1
        )  # (B, 2, d_inner, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B, 4, d_inner, L)

        # Project to get dt, B, C
        xs = xs.permute(0, 1, 3, 2)  # (B, 4, L, d_inner)
        x_dbl = self.x_proj(xs)  # (B, 4, L, dt_rank + 2*d_state)
        dt, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Project dt
        dt = self.dt_proj(dt)  # (B, 4, L, d_inner)
        dt = dt.permute(0, 1, 3, 2)  # (B, 4, d_inner, L)

        # Prepare for selective scan
        xs = xs.permute(0, 1, 3, 2).reshape(B, -1, L)  # (B, 4*d_inner, L)
        dt = dt.reshape(B, -1, L)  # (B, 4*d_inner, L)
        Bs = Bs.reshape(B, K, -1, L)  # (B, 4, d_state, L)
        Cs = Cs.reshape(B, K, -1, L)  # (B, 4, d_state, L)

        # Fix A and D dimensions
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        A = A.repeat(K, 1)  # (4*d_inner, d_state)
        D = self.D.float().repeat(K)  # (4*d_inner,)

        # Perform selective scan
        ys = selective_scan_fn(
            xs, dt, A, Bs, Cs, D,
            delta_softplus=True,
        ).view(B, K, -1, L)  # (B, 4, d_inner, L)

        # Combine scanning directions
        y = ys.sum(dim=1)  # (B, d_inner, L)
        y = y.transpose(1, 2).reshape(B, H, W, -1)  # (B, H, W, d_inner)

        # Output projection
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)

        return y


class MambaBlock(nn.Module):
    def __init__(self, dim, drop_path=0., d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = SS2D(d_model=dim, d_state=d_state)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size

        # SSM branch
        x_2d = x.reshape(B, H, W, C)
        x_ssm = self.norm(x_2d)
        x_ssm = self.ssm(x_ssm)
        x_ssm = x_ssm.reshape(B, L, C)
        x = x + self.drop_path(x_ssm)

        # FFN branch
        x_ffn = self.ffn_norm(x)
        x_ffn = self.ffn(x_ffn)
        x = x + self.drop_path(x_ffn)

        return x


class Mamba(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, depths=(2, 2, 6, 2), d_state=16):
        super().__init__()
        self.embed_dim = embed_dim

        # Input projection
        self.stem = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1, padding=0)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = nn.ModuleList([
                MambaBlock(
                    dim=embed_dim,
                    drop_path=dpr[sum(depths[:i]) + j],
                    d_state=d_state
                ) for j in range(depths[i])
            ])
            self.layers.append(layer)

        # Output projection
        self.output = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)

        self.CA = ChannelAttention(embed_dim, embed_dim)

    def forward(self, x):
        # Input projection
        x = self.stem(x)
        B, C, H, W = x.shape
        x_size = (H, W)

        # Flatten to sequence
        x = x.flatten(2).transpose(1, 2)  # B, L, C

        # Apply Mamba blocks
        for layer in self.layers:
            for blk in layer:
                x = blk(x, x_size)

        # Reshape back to 2D
        x = x.transpose(1, 2).view(B, C, H, W)

        # Output projection
        x = self.output(x)

        x=self.CA(x)


        return x
