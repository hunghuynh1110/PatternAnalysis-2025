import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

# --- Stochastic depth / DropPath (per-sample) ---
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop whole residual paths. (a tiny reimplementation)"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # shape [B, 1, 1, 1] so the same mask is applied to all tokens/channels of each sample
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    
class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        layer_scale_init: Optional[float] = 1e-6,
        norm_type: Literal["ln","bn"] = "ln"
    ):
        super().__init__()
        self.norm_type = norm_type

        # depthwise 7x7 conv (always NCHW here)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)

        hidden_dim = int(dim * mlp_ratio)
        self.act = nn.GELU()

        if norm_type == "ln":
            # channels-last path (original ConvNeXt style)
            self.norm = nn.LayerNorm(dim, eps=1e-6)
            self.pwconv1 = nn.Linear(dim, hidden_dim)
            self.pwconv2 = nn.Linear(hidden_dim, dim)
        else:
            # batch-norm path (NCHW) using 1x1 convs
            self.norm = nn.BatchNorm2d(dim)
            self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True)
            self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)

        # optional LayerScale
        if layer_scale_init is not None:
            self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # depthwise conv (NCHW)
        x = self.dwconv(x)

        if self.norm_type == "ln":
            # switch to channels-last for LN + Linear MLP
            x = x.permute(0, 2, 3, 1)  # NHWC
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2)  # back to NCHW
        else:
            # BN path, stay in NCHW with 1x1 conv MLP
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                # reshape gamma to [C,1,1] to apply per-channel scaling
                x = x * self.gamma.view(1, -1, 1, 1)

        x = shortcut + self.drop_path(x)
        return x
        
        
class ConvNeXtStage(nn.Module):
    def __init__(self, dim, depth, drop_path_rates, downsample=True, norm_type: Literal["ln","bn"]="ln"):
        super().__init__()
        self.downsample = None
        if downsample:
            if norm_type == "ln":
                self.downsample = nn.Sequential(
                    nn.LayerNorm(dim, eps=1e-6),
                    nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2)
                )
            else:
                self.downsample = nn.Sequential(
                    nn.BatchNorm2d(dim),
                    nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2)
                )
            dim *= 2

        blocks = []
        for i in range(depth):
            blocks.append(
                ConvNeXtBlock(
                    dim=dim,
                    drop_path=drop_path_rates[i] if isinstance(drop_path_rates, list) else drop_path_rates,
                    norm_type=norm_type
                )
            )

        self.blocks = nn.Sequential(*blocks)
        self.out_dim = dim

    def forward(self, x):
        if self.downsample is not None:
            # If LayerNorm was used in downsample[0], it expects NHWC.
            if isinstance(self.downsample[0], nn.LayerNorm):
                x = x.permute(0, 2, 3, 1)
                x = self.downsample[0](x)
                x = x.permute(0, 3, 1, 2)
                x = self.downsample[1](x)
            else:
                # BatchNorm2d path (stays NCHW)
                x = self.downsample[0](x)
                x = self.downsample[1](x)
        return self.blocks(x)

class ConvNeXtMRI(nn.Module):
    def __init__(self, in_chans=3, num_classes=2, depths=[2,2,4,2],
                 dims=[48,96,192,384],
                 drop_path_rate=0.1,
                 norm_type: Literal["ln","bn"] = "ln"):
        super().__init__()
        self.norm_type = norm_type

        # Stem: patchify image
        if norm_type == "ln":
            self.stem_conv = nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4)
            self.stem_norm = nn.LayerNorm(dims[0], eps=1e-6)
        else:
            self.stem_conv = nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4)
            self.stem_norm = nn.BatchNorm2d(dims[0])

        # drop path schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stage
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = ConvNeXtStage(
                dim=dims[i] if i == 0 else dims[i-1],
                depth=depths[i],
                drop_path_rates=dpr[cur:cur + depths[i]],
                downsample=(i != 0),
                norm_type=norm_type
            )
            self.stages.append(stage)
            cur += depths[i]

        # final normalization & classifier
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        ## init weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem_conv(x)
        if isinstance(self.stem_norm, nn.LayerNorm):
            x = x.permute(0, 2, 3, 1)
            x = self.stem_norm(x)
            x = x.permute(0, 3, 1, 2)
        else:
            x = self.stem_norm(x)

        for stage in self.stages:
            x = stage(x)

        # Global average pooling (mean over H and W)
        x = x.mean([-2, -1])
        x = self.norm(x)
        x = self.head(x)
        return x
        
        
        
        
        
        
        
        
        

if __name__ == "__main__":
    model = ConvNeXtMRI()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)