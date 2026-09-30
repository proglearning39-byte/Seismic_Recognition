# model.py
import torch
import torch.nn as nn

class PatchTSTEncoder(nn.Module):
    def __init__(self, seq_len=1600, patch_length=50, stride=25, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        self.num_patches = (seq_len - patch_length) // stride + 1
        
        # 匹配 Unexpected Keys: model.encoder.embedder...
        self.embedder = nn.ModuleDict({
            'input_embedding': nn.Linear(patch_length, d_model)
        })
        self.positional_encoder = nn.ModuleDict({
            'position_enc': nn.Parameter(torch.randn(1, self.num_patches, d_model))
        })
        
        # 匹配 Encoder 多層結構
        layers = []
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, nhead, batch_first=True),
                'norm_sublayer1': nn.ModuleDict({'batchnorm': nn.BatchNorm1d(d_model)}),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(d_model * 2, d_model)
                ),
                'norm_sublayer3': nn.ModuleDict({'batchnorm': nn.BatchNorm1d(d_model)})
            })
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

class PatchTSTEEWRobust(nn.Module):
    def __init__(self, seq_len=1600, patch_length=50, stride=25, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        
        # 包裹一層 self.model 以完全匹配 .pt 檔的 model.encoder... Key
        self.model = nn.ModuleDict({
            'encoder': PatchTSTEncoder(seq_len, patch_length, stride, d_model, nhead, num_layers)
        })
        
        num_patches = (seq_len - patch_length) // stride + 1
        self.head = nn.ModuleDict({
            'linear': nn.Linear(num_patches * d_model, 1)
        })

    def forward(self, x):
        # x: (batch_size, 1, seq_len)
        x = x.squeeze(1)
        
        # Patchify
        patches = x.unfold(dimension=1, size=self.patch_length, step=self.stride)
        
        # Embedding
        enc = self.model['encoder']
        x_emb = enc.embedder['input_embedding'](patches) + enc.positional_encoder['position_enc']
        
        # Transformer Custom Forward
        for layer in enc.layers:
            # Self Attention
            attn_out, _ = layer['self_attn'](x_emb, x_emb, x_emb)
            x_emb = x_emb + attn_out
            
            # BatchNorm 1 (Transpose for Channel-first BatchNorm)
            x_norm = layer['norm_sublayer1']['batchnorm'](x_emb.transpose(1, 2)).transpose(1, 2)
            
            # FeedForward
            ff_out = layer['ff'](x_norm)
            x_emb = x_norm + ff_out
            x_emb = layer['norm_sublayer3']['batchnorm'](x_emb.transpose(1, 2)).transpose(1, 2)
            
        # Head
        x_flat = x_emb.reshape(x_emb.size(0), -1)
        logits = self.head['linear'](x_flat)
        return logits
