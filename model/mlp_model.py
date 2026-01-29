import torch
import torch.nn as nn
from model.sf_model import SFDecoder


class MLP_MLP(nn.Module):

    def __init__(self, input_dim=56, latent_dim=32, hidden_dim=512):
        super(MLP_MLP, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Bottleneck layer
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # Reconstruct original parameters
        )


    def forward(self, x):
        latent = self.encoder(x)
        mu, log_var = latent.chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var) + 1e-8
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick
        reconstructed = self.decoder(z)
        return reconstructed, z, mu, log_var


    def encode(self, x):
        latent = self.encoder(x)
        mu, log_var = latent.chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var) + 1e-8
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick
        return z, mu, log_var


class MLP_SF(nn.Module):

    def __init__(self, input_dim=56, latent_dim=32, hidden_dim=256, grid_dim=24, deterministic=False):
        super(MLP_SF, self).__init__()

        self.deterministic = deterministic

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # For mean and log variance
        )

        # Decoder
        self.decoder = SFDecoder(grid_dim=grid_dim, out_dim=7, feature_dim=latent_dim)

    def forward(self, x):
        params = self.encoder(x)
        mu, log_var = params.chunk(2, dim=-1)
        if self.deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, log_var)
        reconstructed = self.decoder(z)
        return reconstructed, z, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) + 1e-8
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        params = self.encoder(x)
        mu, log_var = params.chunk(2, dim=-1)
        if self.deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, log_var)
        return z, mu, log_var
