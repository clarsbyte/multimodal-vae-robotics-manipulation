import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from models.utils import PositionalEncoding, product_of_experts
from models.decoders import Dec_Image, Dec_Text, Dec_Action
from models.encoders import Enc_Image, Enc_Text, Enc_Action

class MMVAE(nn.Module):
    def __init__(self,latent_dim=32, vocab_size=16, action_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_image = Enc_Image(latent_dim)
        self.enc_txt = Enc_Text(vocab_size, latent_dim)
        self.enc_act = Enc_Action(action_dim=action_dim, latent_dim=latent_dim)

        self.dec_image = Dec_Image(latent_dim, hidden_dim=512)
        self.dec_txt = Dec_Text(latent_dim, vocab_size=vocab_size)
        self.dec_act = Dec_Action(latent_dim, action_dim)

    def infer(self,batch): # inference
        b = batch["batch_size"]
        dev = next(self.parameters()).device

        # stacked as (num_experts, B, latent_dim)
        mu_list = [torch.zeros(1, b, self.latent_dim, device=dev)]
        logvar_list = [torch.zeros(1, b, self.latent_dim, device=dev)]

        if batch.get("image") is not None:
            m, lv = self.enc_image(batch["image"])
            mu_list.append(m.unsqueeze(0)); logvar_list.append(lv.unsqueeze(0))
        if batch.get("text") is not None:
            m, lv = self.enc_txt(batch["text"], batch["text_mask"])
            mu_list.append(m.unsqueeze(0)); logvar_list.append(lv.unsqueeze(0))
        if batch.get("action") is not None:
            m, lv = self.enc_act(batch["action"], batch["action_mask"])
            mu_list.append(m.unsqueeze(0)); logvar_list.append(lv.unsqueeze(0))


        return product_of_experts(torch.cat(mu_list,0), torch.cat(logvar_list,0))   

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std

    def forward(self,batch):
        mu, logvar = self.infer(batch)
        z = self.reparameterize(mu, logvar)

        return {
            "image":  self.dec_image(z),
            "text":   self.dec_txt(z, batch["text_mask"]),
            "action": self.dec_act(z, batch["action_mask"]),
            "mu": mu, "logvar": logvar,
        }