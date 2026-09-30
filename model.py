import torch
import torch.nn as nn
import math

class SelfAttentionCustom(nn.Module):
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # 精準對應 q_proj, k_proj, v_proj, out_proj
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = torch.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)

class PatchTSTEncoderLayer(nn.Module):
    def __init__(self, d_model=64, nhead=4, d_ff=512):
        super().__init__()
        self.self_attn = SelfAttentionCustom(d_model=d_model, nhead=nhead)
        
        # 匹配 norm_sublayer1.batchnorm
        self.norm_sublayer1 = nn.ModuleDict({
            'batchnorm': nn.BatchNorm1d(d_model)
        })
        
        # 匹配 ff.0 (Linear 64->512), ff.3 (Linear 512->64)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model)
        )
        
        # 匹配 norm_sublayer3.batchnorm
        self.norm_sublayer3 = nn.ModuleDict({
            'batchnorm': nn.BatchNorm1d(d_model)
        })

    def forward(self, x):
        attn_out = self.self_attn(x)
        x = x + attn_out
        
        # Channel-first BatchNorm
        x = self.norm_sublayer1['batchnorm'](x.transpose(1, 2)).transpose(1, 2)
        
        ff_out = self.ff(x)
        x = x + ff_out
        x = self.norm_sublayer3['batchnorm'](x.transpose(1, 2)).transpose(1, 2)
        return x

class PatchTSTEncoder(nn.Module):
    def __init__(self, num_patches=1951, patch_length=16, d_model=64, nhead=4, num_layers=3, d_ff=512):
        super().__init__()
        # 匹配 embedder.input_embedding
        self.embedder = nn.ModuleDict({
            'input_embedding': nn.Linear(patch_length, d_model)
        })
        
        # 匹配 positional_encoder.position_enc [1951, 64]
        class PositionalEncodingModule(nn.Module):
            def __init__(self, num_patches, d_model):
                super().__init__()
                self.position_enc = nn.Parameter(torch.randn(num_patches, d_model))
        self.positional_encoder = PositionalEncodingModule(num_patches, d_model)
        
        self.layers = nn.ModuleList([
            PatchTSTEncoderLayer(d_model=d_model, nhead=nhead, d_ff=d_ff) for _ in range(num_layers)
        ])

    def forward(self, patches):
        x_emb = self.embedder['input_embedding'](patches) + self.positional_encoder.position_enc
        for layer in self.layers:
            x_emb = layer(x_emb)
        return x_emb

class PatchTSTEEWRobust(nn.Module):
    def __init__(self, seq_len=1600, patch_length=16, stride=1, d_model=64, nhead=4, num_layers=3, d_ff=512):
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        
        # 自動根據 num_patches=1951 調整
        num_patches = 1951
        
        self.model = nn.ModuleDict({
            'encoder': PatchTSTEncoder(num_patches=num_patches, patch_length=patch_length, d_model=d_model, nhead=nhead, num_layers=num_layers, d_ff=d_ff)
        })
        
        # 匹配 head.linear [2, 64]
        self.head = nn.ModuleDict({
            'linear': nn.Linear(d_model, 2)
        })

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
            
        # 補足序列長度以符合 1951 個 patch (若輸入不足)
        # Patchify unfold
        patches = x.unfold(dimension=1, size=self.patch_length, step=self.stride)
        if patches.size(1) < 1951:
            pad_len = 1951 - patches.size(1)
            patches = torch.nn.functional.pad(patches, (0, 0, 0, pad_len))
        elif patches.size(1) > 1951:
            patches = patches[:, :1951, :]
            
        x_trans = self.model['encoder'](patches)
        
        # Global Average Pooling 取最後特徵維度 -> [B, 64]
        x_pool = x_trans.mean(dim=1)
        logits = self.head['linear'](x_pool)
        return logits
