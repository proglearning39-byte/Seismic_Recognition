import torch
import torch.nn as nn

class Patchify(nn.Module):
    def __init__(self, patch_length=50, stride=25): # 0.5s patch, 50% overlap
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride

    def forward(self, x):
        # x: (batch_size, 1, seq_len)
        x = x.squeeze(1)
        patches = x.unfold(dimension=1, size=self.patch_length, step=self.stride)
        return patches

class PatchTSTEEWRobust(nn.Module):
    def __init__(self, seq_len=1600, patch_length=50, stride=25, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.patchify = Patchify(patch_length=patch_length, stride=stride)
        num_patches = (seq_len - patch_length) // stride + 1
        
        self.patch_embedding = nn.Linear(patch_length, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(num_patches * d_model),
            nn.Dropout(0.3),
            nn.Linear(num_patches * d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Logits 輸出
        )

    def forward(self, x):
        patches = self.patchify(x)
        x_emb = self.patch_embedding(patches) + self.pos_embedding
        x_trans = self.transformer(x_emb)
        logits = self.head(x_trans)
        return logits
