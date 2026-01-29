import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sf_model import ChamferLoss, SFModel
from model.mlp_model import MLP_MLP, MLP_SF


import warnings
warnings.filterwarnings("ignore")


class SFVAE(nn.Module):

    def __init__(self, embedding_dim=32, grid_dim=24, vae=True, norm_weight=0, deterministic=False):
        super(SFVAE, self).__init__()
        self.is_vae = vae
        self.model = SFModel(input_dim=7, feature_dim=embedding_dim, grid_dim=grid_dim, 
                             vae=vae, deterministic=deterministic)
        self.criterion = ChamferLoss(geo_weight=0.5)
        self.norm_weight = norm_weight


    def forward(self, data):
        input = data
        if self.is_vae:
            output, z, mu, logvar = self.model(input)
            chamfer_loss, _, _ =  self.criterion(input, output)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / input.size(0)
            loss = chamfer_loss + self.norm_weight * kl_loss
            return loss, output, z
        else:
            output, embedding = self.model(input)
            chamfer_loss, loss_geo, loss_color = self.criterion(input, output)
            return loss_geo, loss_color, output, embedding


    def encode(self, input):
        if self.is_vae:
            embedding, _, _ = self.model.encoder(input)
        else:
            embedding = self.model.encoder(input)
        return embedding


    def decode(self, embedding):
        return self.model.decoder(embedding)


class ParamMLP(nn.Module):

    def __init__(self, embedding_dim=32, hidden_dim=512, decoder_type="sf", grid_dim=24, norm_weight=1e-4, deterministic=False):
        super(ParamMLP, self).__init__()
        assert decoder_type in ["sf", "mlp"], "decoder_type must be 'sf' or 'mlp'"
        self.decoder_type = decoder_type
        if self.decoder_type == "sf":
            self.model = MLP_SF(input_dim=56, latent_dim=embedding_dim, hidden_dim=hidden_dim, 
                                      grid_dim=grid_dim, deterministic=deterministic)
            self.criterion = ChamferLoss(geo_weight=0.5)
        else:
            self.model = MLP_MLP(input_dim=56, latent_dim=embedding_dim, hidden_dim=hidden_dim)
            self.criterion = nn.L1Loss()
        self.norm_weight = norm_weight


    def forward(self, data):
        if self.decoder_type == "sf":
            param, gt = data
        else:
            param = data

        output, z, mu, logvar = self.model(param)

        output = torch.clamp(output, min=-10, max=10)
        mu = torch.clamp(mu, min=-10, max=10)
        logvar = torch.clamp(logvar, min=-10, max=10)

        if self.decoder_type == "sf":
            chamfer_loss, _, _ =  self.criterion(gt, output)
            loss_metric = chamfer_loss
        else:
            l1_loss = self.criterion(param, output)
            loss_metric = l1_loss

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / param.size(0)

        # numerical safeguards
        if torch.isnan(loss_metric) or torch.isinf(loss_metric):
            loss_metric = torch.tensor(0.0, device=param.device)
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            kl_loss = torch.tensor(0.0, device=param.device)

        loss = loss_metric + self.norm_weight * kl_loss

        return loss, output, z


    def encode(self, input):
        embedding, _, _ = self.model.encode(input)
        return embedding


    def decode(self, embedding):
        return self.model.decoder(embedding)


def downsample_points(pred, num_points=12*12):
    pred = pred.permute(0, 2, 1)
    pred = F.adaptive_avg_pool1d(pred, num_points)
    pred = pred.permute(0, 2, 1)
    return pred


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
