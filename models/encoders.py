import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from models.utils import PositionalEncoding
# from models import AutoEncoder

# class VAE(AutoEncoder):
#     def __init__(self,input_dim,hidden_dim,latent_dim,device):
#         super(VAE,self).__init__(input_dim,hidden_dim,latent_dim)
#         self.mu = nn.Linear(hidden_dim, latent_dim)
#         self.logvar = nn.Linear(hidden_dim, latent_dim)
#         self.device = device

#         self.encoder_vae = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU()
#         )

#         del self.encoder # Remove the original encoder from AutoEncoder because we want the latent normal dist in between hidden and latent space

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std) #[0,1] with size of std
#         return mu + eps * std
    
#     def forward(self,x):
#         encoded = self.encoder_vae(x)
#         mu = self.mu(encoded)
#         logvar = self.logvar(encoded)
#         z = self.reparameterize(mu, logvar)
#         decoded = self.decoder(z)
#         return encoded, decoded, mu, logvar
    
#     def sample(self, num_samples):
#         with torch.no_grad():
#             z = torch.randn(num_samples, self.latent_dim).to(self.device)
#             samples = self.decoder(z)
#         return samples

        
#     # Binary cross-entropy and Kullback-Leibler divergence
#     def loss_function(recon_x, x, mu, logvar):
#         BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
#         KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#         return BCE + KLD


def masked_mean(h, mask):
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1)

class Enc_Image(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()
        self.mu  = nn.Linear(2048, latent_dim)
        self.logvar = nn.Linear(2048, latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        return self.mu(h), self.logvar(h)   

class Enc_Text(nn.Module):
    # d model is like the dimension per token 
    def __init__(self, vocab_size, latent_dim,  d_model=32, ff=128, layers=1, heads=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,d_model,padding_idx=0)
        self.pos = PositionalEncoding(d_model)

        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout=0.1, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

        self.mu = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)

    def forward(self, x, mask): # encoder only
        h = self.pos(self.embed(x))
        h = self.encoder(h, src_key_padding_mask=~mask)
        h = masked_mean(h, mask)
        return self.mu(h), self.logvar(h)

class Enc_Action(nn.Module):
    def __init__(self, action_dim=4, latent_dim=64,d_model=64,ff=1024,layers=8,heads=2,dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(action_dim, d_model)
        self.pos = PositionalEncoding(d_model)

        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout=dropout, activation="gelu", batch_first=True)

        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.mu = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)

    def forward(self, x, mask):
        h = self.pos(self.embed(x))
        h = self.encoder(h, src_key_padding_mask=~mask)
        h = masked_mean(h, mask)
    
        return self.mu(h), self.logvar(h)