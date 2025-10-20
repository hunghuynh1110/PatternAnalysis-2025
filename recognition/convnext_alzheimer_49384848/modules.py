import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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
        layer_scale_init: Optional[float] = 1e-6
    ):
        super().__init__()
        
        #depthwise 7x7 conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding = 3, groups = dim, bias = True)
        
        #channels_last layerNorm
        self.norm = nn.LayerNorm(dim, eps = 1e-6)
        
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        
        #optional LayerScale
        if layer_scale_init is not None:
            self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        else:
            self.gamma = None
            
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shorcut = x
        
        #depthwise conv in NCHW
        x = self.dwconv(x)  #[Bsize, Channels, Height, Width]
        
        #switch to channels-last for LayerNorm + MLP
        x = x.permute(0,2,3,1)  #rearrange
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
        
        x = x.permute(0,3,1,2)
        
        x = shorcut + self.drop_path(x)
        return x
        
        
class ConvNeXtStage(nn.Module):
    def __init__(self, dim, depth, drop_path_rates, downsample = True):
        super().__init__()
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(nn.LayerNorm(dim, eps=1e-6), nn.Conv2d(dim, dim*2, kernel_size=2, stride = 2))
            dim *=2
            
        blocks = []
        for i in range(depth):
            blocks.append(
                ConvNeXtBlock(
                    dim=dim, 
                    drop_path=drop_path_rates[i] if isinstance(drop_path_rates,list) else drop_path_rates
                )
            )
            
        self.blocks = nn.Sequential(*blocks)
        self.out_dim = dim
    
    def forward(self,x):
        if self.downsample is not None:
            x = self.downsample[0](x.permute(0,2,3,1))
            x = x.permute(0,3,1,2)
            x = self.downsample[1](x)
        return self.blocks(x)

class ConvNeXtMRI(nn.Module):
    def __init__(self, in_chans=3, num_classes = 2, depths = [2,2,4,2],
                 dims = [48,96,192,384],
                 drop_path_rate = 0.1):
        super().__init__()
        
        #Stem: patchify image
        self.stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride = 4),
                                  nn.LayerNorm(dims[0], eps=1e-6))
        
        #drop path schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build stage
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = ConvNeXtStage(
                dim = dims[i] if i ==0 else dims[i-1],
                depth=depths[i],
                drop_path_rates=dpr[cur:cur + depths[i]],
                downsample=(i != 0)
            )
            self.stages.append(stage)
            cur += depths[i]
            
        
        # final normalization & classifier
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        
        ## init weight
        self.apply(self._init_weights)
        
    
    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std = .02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    
    def forward(self,x):
        x = self.stem[0](x)
        x = x.permute(0, 2, 3, 1)
        x = self.stem[1](x)
        x = x.permute(0, 3, 1, 2)

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