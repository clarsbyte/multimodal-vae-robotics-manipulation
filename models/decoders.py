import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from models.utils import PositionalEncoding
"""
Linear(latent_dim → 512) → ReLU
Linear(512 → 512) → ReLU
Linear(512 → 32*4*4) → ReLU
reshape to (B, 32, 4, 4)
ConvTranspose2d(32→32, k=4, s=2, p=1) →  8×8 → ReLU
ConvTranspose2d(32→32, k=4, s=2, p=1) → 16×16 → ReLU
ConvTranspose2d(32→32, k=4, s=2, p=1) → 32×32 → ReLU
ConvTranspose2d(32→3,  k=4, s=2, p=1) → 64×64
sigmoid
"""
class Dec_Image(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32 * 4 * 4),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),   
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),   
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 32, 4, 4)  # Reshape to (batch_size, channels, height, width)
        return self.deconv(z)

class Dec_Text(nn.Module):
    def __init__(self, latent_dim=64, vocab_size=16, d_model=32, ff=128,layers=1, heads=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(latent_dim, d_model)
        self.pos = PositionalEncoding(d_model, dropout)
        layer = nn.TransformerDecoderLayer(d_model, heads, ff, dropout=dropout,activation="gelu", batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, z, mask): 
        B, L = mask.shape
        memory = self.proj(z).unsqueeze(1)
        q = self.pos(torch.zeros(B, L, self.d_model, device=z.device))
        h = self.decoder(tgt=q, memory=memory, tgt_key_padding_mask=~mask)
        return self.out(h) 

class Dec_Action(nn.Module):
    def __init__(self, latent_dim=64, action_dim=4, d_model=64, ff=1024,layers=4, heads=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(latent_dim, d_model)
        self.pos = PositionalEncoding(d_model, dropout)
        layer = nn.TransformerDecoderLayer(d_model, heads, ff, dropout=dropout, activation="gelu", batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.out = nn.Linear(d_model, action_dim)

    def forward(self, z, mask):                
        B, T = mask.shape
        memory = self.proj(z).unsqueeze(1)
        q = torch.zeros(B, T, self.d_model, device=z.device)
        q = self.pos(q)
        h = self.decoder(tgt=q, memory=memory, tgt_key_padding_mask=~mask)
        return self.out(h) * mask.unsqueeze(-1) 
