import torch
import torch.nn as nn

class PatchTSTEncoderLayer(nn.Module):
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        # 匹配 key: norm_sublayer1.batchnorm
        self.norm_sublayer1 = nn.ModuleDict({
            'batchnorm': nn.BatchNorm1d(d_model)
        })
        # 匹配 key: ff.0 (Linear), ff.3 (Linear)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        # 匹配 key: norm_sublayer3.batchnorm
        self.norm_sublayer3 = nn.ModuleDict({
            'batchnorm': nn.BatchNorm1d(d_model)
        })

    def forward(self, x):
        # Self-Attention
        attn_out, _ = self.self_attn(x, x, x)
        x = x + attn_out
        
        # BatchNorm 1 (BatchNorm1d 需切換為 N, C, L)
        x_norm = self.norm_sublayer1['batchnorm'](x.transpose(1, 2)).transpose(1, 2)
        
        # FeedForward
        ff_out = self.ff(x_norm)
        x = x_norm + ff_out
        x = self.norm_sublayer3['batchnorm'](x.transpose(1, 2)).transpose(1, 2)
        return x

class PatchTSTEncoder(nn.Module):
    def __init__(self, seq_len=1600, patch_length=50, stride=25, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        self.num_patches = (seq_len - patch_length) // stride + 1
        
        # 匹配 key: model.encoder.embedder.input_embedding
        self.embedder = nn.ModuleDict({
            'input_embedding': nn.Linear(patch_length, d_model)
        })
        # 匹配 key: model.encoder.positional_encoder
        # 注意：將 Tensor 包裹於 Module 內部以符合 nn.ModuleDict 規範
        class PositionalEncodingModule(nn.Module):
            def __init__(self, num_patches, d_model):
                super().__init__()
                self.position_enc = nn.Parameter(torch.randn(1, num_patches, d_model))
        self.positional_encoder = PositionalEncodingModule(self.num_patches, d_model)
        
        # 匹配 key: model.encoder.layers.0, 1, 2...
        self.layers = nn.ModuleList([
            PatchTSTEncoderLayer(d_model=d_model, nhead=nhead) for _ in range(num_layers)
        ])

    def forward(self, patches):
        x_emb = self.embedder['input_embedding'](patches) + self.positional_encoder.position_enc
        for layer in self.layers:
            x_emb = layer(x_emb)
        return x_emb

class PatchTSTEEWRobust(nn.Module):
    def __init__(self, seq_len=1600, patch_length=50, stride=25, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        
        # 匹配 key: model.encoder...
        self.model = nn.ModuleDict({
            'encoder': PatchTSTEncoder(seq_len, patch_length, stride, d_model, nhead, num_layers)
        })
        
        num_patches = (seq_len - patch_length) // stride + 1
        # 匹配 key: head.linear...
        self.head = nn.ModuleDict({
            'linear': nn.Linear(num_patches * d_model, 1)
        })

    def forward(self, x):
        # x: (batch_size, 1, seq_len)
        if x.dim() == 3:
            x = x.squeeze(1)
            
        # Patchify
        patches = x.unfold(dimension=1, size=self.patch_length, step=self.stride)
        
        # Forward pass through encoder
        x_trans = self.model['encoder'](patches)
        
        # Flatten head
        x_flat = x_trans.reshape(x_trans.size(0), -1)
        logits = self.head['linear'](x_flat)
        return logits
