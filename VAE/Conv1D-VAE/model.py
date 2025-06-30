import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
num_features = 122
seq_len = 30
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return self.relu(out + identity)

class ConvTimeSeriesVAE(nn.Module):
    def __init__(self, input_dim=122, latent_dim=64, hidden_dim=128, seq_len=60):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.Sequential(
            ResidualBlock1D(input_dim, hidden_dim),
            ResidualBlock1D(hidden_dim, hidden_dim),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim * seq_len)
        self.decoder = nn.Sequential(
            ResidualBlock1D(hidden_dim, hidden_dim),
            ResidualBlock1D(hidden_dim, hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.permute(0, 2, 1)                
        x = self.encoder(x)                   
        h = x.squeeze(-1)                     
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)                          
        x = x.view(-1, self.hidden_dim, self.seq_len)
        x = self.decoder(x)                             
        x = x.transpose(1, 2)                           
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl, recon_loss, kl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvTimeSeriesVAE(input_dim=num_features, hidden_dim=64, latent_dim=32, seq_len=seq_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)