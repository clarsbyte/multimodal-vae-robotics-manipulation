import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   

    def forward(self, x):                           
        return self.dropout(x + self.pe[:, :x.size(1)])

# multiply the densities together
def product_of_experts(mu, logvar, eps=1e-8):
    var = torch.exp(logvar) + eps
    T = 1.0 / var              
    pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
    pd_var = 1.0 / torch.sum(T, dim=0)
    return pd_mu, torch.log(pd_var)        

def softclip(t, min_val=-6.0):
    # soft lower bound on log_sigma, as in sigma-vae-pytorch
    return min_val + F.softplus(t - min_val)

def sigma_vae_loss(recon, target, mask=None):
    if mask is not None:
        m = mask.unsqueeze(-1).float()
        mse = ((recon - target) ** 2 * m).sum() / (m.sum() * target.shape[-1])
    else:
        m = None
        mse = F.mse_loss(recon, target)
    log_sigma = softclip(0.5 * torch.log(mse + 1e-8))
    nll = 0.5 * ((recon - target) / log_sigma.exp()) ** 2 + log_sigma + 0.5 * math.log(2 * math.pi)
    if m is not None:
        nll = nll * m
    return nll.sum() / target.shape[0]

def masked_cross_entropy(logits, target_ids, mask):
    ce = F.cross_entropy(logits.transpose(1, 2), target_ids, reduction="none")
    return (ce * mask.float()).sum() / target_ids.shape[0]

def kl_divergence(mu, logvar):
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()