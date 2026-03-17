from dataset import GaussianGen
from dataset import Ply
from torch.utils.data import DataLoader
import torch

from embedding_model import SFVAE, ParamMLP

import tqdm
from utils.log import log_csv
from utils.visualize import visualize_point_cloud
import numpy as np
import random
import os
import yaml

import warnings
warnings.filterwarnings("ignore")

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "train.yaml")


def save_checkpoint(state, filename="checkpoint.pth"):
    print(f"Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(filename="checkpoint.pth"):
    print(f"Loading embedding model checkpoint from {filename}")
    return torch.load(filename, map_location="cuda")


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    np.random.seed(seed)
    random.seed(seed)

    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_train_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Training config at {config_path} must be a YAML mapping.")

    return config


def train(
        model_type="sfvae", 
        dataset="gaussiangen", 
        dataset_path=None,
        num_points=12*12, 
        num_samples=100000, 
        epoch=1000, 
        bs=1000, 
        embedding_dim=32, 
        norm_weight=1e-4, 
        grid_dim=24, 
        cuda=0, 
        save_model=True, 
        log=True, 
        weight_path="checkpoints/checkpoint.pth", 
        log_path="log/training_log.csv",
        validation=False,
        resume=False
    ):

    if dataset == "gaussiangen":
        dataset = GaussianGen(num_samples=num_samples, num_points=num_points, 
                              max_scale=0, min_scale=-8, sh_degree=0,
                              return_type="both" if model_type == "mlp" else "gaussian")
    else:
        if dataset_path is None:
            raise ValueError("`dataset_path` must be provided when using `--dataset ply`.")
        dataset = Ply(num_points=num_points, path=dataset_path, random_choose=0.0005,
                      return_type="both" if model_type == "mlp" else "gaussian")
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, 
                            num_workers=24, pin_memory=True, prefetch_factor=4)

    if model_type == "sfvae":
        model = SFVAE(embedding_dim=embedding_dim, grid_dim=grid_dim, norm_weight=norm_weight)
    else:
        model = ParamMLP(embedding_dim=embedding_dim, hidden_dim=256, 
                         decoder_type="mlp", norm_weight=norm_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-6)

    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Current device set to: {device}")

    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    model = torch.compile(model)
    model.to(device)

    start_epoch = 0
    if resume and weight_path is not None and os.path.exists(weight_path):
        checkpoint = load_checkpoint(weight_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1  # Resume from the next epoch
        print(f"Resuming training from epoch {start_epoch}")

    model.train()

    iter = 0
    loss_all, loss_geo_all, loss_color_all = 0, 0, 0
    for e in range(start_epoch, epoch):  # Start from the resumed epoch
        print(f"Epoch: {e}, lr: {scheduler.get_last_lr()}")
        # train for 1 epoch
        bar = tqdm.tqdm(total=len(dataloader))
        for i, data in enumerate(dataloader):
            if type(data) is list:
                data = [d.float().to(device) for d in data]
            else:
                data = data.float().to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(enabled=torch.cuda.is_available(), device_type='cuda'):
                if model_type == "sfvae":
                    loss, _, _ = model(data)
                else:
                    points, param = data
                    param = param[..., 3:]  # remove centroid
                    input_data = (param, points) if model.decoder_type == "sf" else param
                    loss, _, _ = model(input_data)
                loss_all += loss.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            bar.update(1)
            iter += 1
            bar.set_description(f"iter: {iter}, loss: {round(loss.item(), 5)}")

        bar.close()

        loss_all /= len(dataloader)
        loss_geo_all /= len(dataloader)
        loss_color_all /= len(dataloader)
        print(f"Avg loss: {loss_all}, Avg geo loss: {loss_geo_all}, Avg color loss: {loss_color_all}")

        scheduler.step()

        if log:
            log_csv(log_path, e, loss_all, loss_geo_all, loss_color_all)
        
        if validation:
            if e % 10 == 0: # validate every 10 epochs
                visual_validation(model, dataloader, model_type)

        if save_model:
            save_checkpoint({
                "epoch": e,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": loss
            }, filename=weight_path)


def visual_validation(model, dataloader, model_type):
    """visualize gt vs pred, validate during training"""

    data = next(iter(dataloader))
    if type(data) is list:
        data = [d.float().to(next(model.parameters()).device) for d in data]
    else:
        data = data.float().to(next(model.parameters()).device)
    
    with torch.no_grad():
        model.eval()
        if model_type == "sfvae":
            _, pred, _ = model(data)
            gt = data
        else:
            points, param = data
            param = param[..., 3:]  # remove centroid
            input_data = (param, points) if model.decoder_type == "sf" else param
            _, pred, _ = model(input_data)
            gt = param

        # visualize first element in the batch
        visualize_point_cloud(pred, filename="pred.png")
        visualize_point_cloud(gt, filename="gt.png")

        model.train()


if __name__ == "__main__":
    import argparse

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='Path to training config YAML')
    config_args, remaining_argv = config_parser.parse_known_args()
    config = load_train_config(config_args.config)

    parser = argparse.ArgumentParser(description="Tokenizer Training and Visualization", parents=[config_parser])
    parser.set_defaults(**config)
    # training parameters
    parser.add_argument('--model', type=str, default='sfvae', choices=['sfvae', 'mlp'], help='Model type')
    parser.add_argument('--dataset', type=str, default='gaussiangen', choices=['gaussiangen', 'ply'], help='Dataset type')
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to the PLY dataset root when using --dataset ply')
    parser.add_argument('--num_points', type=int, default=12*12, help='Number of points')
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of samples')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=1000, help='Batch size')
    parser.add_argument('--embedding_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--norm_weight', type=float, default=0, help='Normalization weight')
    parser.add_argument('--grid_dim', type=int, default=24, help='Grid dimension for SF-VAE')

    parser.add_argument('--cuda', type=int, default=0, help='Visible CUDA device')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--save_model', type=bool, default=True, help='Save model')
    parser.add_argument('--log', type=bool, default=True, help='Log training')
    parser.add_argument('--weight_path', type=str, default='checkpoints/checkpoint.pth', help='Path to model weights')
    parser.add_argument('--log_path', type=str, default='log/training_log.csv', help='Path to save logs')
    parser.add_argument('--validation', type=bool, default=False, help='Whether to do visual validation during training')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training from checkpoint')

    args = parser.parse_args(remaining_argv)

    if args.seed is not None:
        set_seed(args.seed) # set seed for reproducibility

    train(
        model_type=args.model,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        num_points=args.num_points,
        num_samples=args.num_samples,
        epoch=args.epoch,
        bs=args.bs,
        embedding_dim=args.embedding_dim,
        norm_weight=args.norm_weight,
        grid_dim=args.grid_dim,
        cuda=args.cuda,
        save_model=args.save_model,
        log=args.log,
        weight_path=args.weight_path,
        log_path=args.log_path,
        validation=args.validation,
        resume=args.resume
    )
