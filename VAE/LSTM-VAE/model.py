import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
num_features = 122
seq_len = 30
import os


class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.tanh_out = nn.Tanh()

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h = h_n[-1]
        mu = self.fc_mu(h)        
        logvar = self.fc_logvar(h) 
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        hidden = self.decoder_input(z).unsqueeze(1).repeat(1, seq_len, 1)
        dec_out, _ = self.decoder_lstm(hidden)
        out = self.output_layer(dec_out)
        out = self.tanh_out(out)   # áp tanh ở cuối output
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, x.size(1))
        return recon_x, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl
def kl_annealing(epoch, total_anneal_epochs=50, max_beta=1.0):
    if epoch < total_anneal_epochs:
        return max_beta * (epoch / total_anneal_epochs)
    else:
        return max_beta


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMVAE(input_dim=num_features, hidden_dim1=64, hidden_dim2=32, latent_dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)