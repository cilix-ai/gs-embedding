import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from converter import Converter
from utils.visualize import visualize_gaussian
from dataset import Gaussian
from utils.eval_metrics import *


class FourierEncoding(nn.Module):
    def __init__(self, num_bands=10, max_freq=10.0, include_input=True):
        super().__init__()
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.include_input = include_input
        freqs = np.logspace(0.0, np.log10(max_freq), num_bands)
        self.register_buffer("freqs", torch.tensor(freqs, dtype=torch.float32))

    def forward(self, x):
        # x: (..., 3)
        x_exp = [x] if self.include_input else []
        for f in self.freqs:
            # print(x.device, f.device)
            x_exp.append(torch.sin(x * f))
            x_exp.append(torch.cos(x * f))
        return torch.cat(x_exp, dim=-1)


class ProgressiveFourierEncoding(nn.Module):
    def __init__(self, freqs: torch.Tensor, include_input=True):
        super().__init__()
        self.register_buffer("freqs", freqs)
        self.include_input = include_input
        self.progress = 1.0  # fraction of bands active (0..1)

    def set_progress(self, ratio: float):
        self.progress = float(max(0.0, min(1.0, ratio)))

    def forward(self, x):
        # Always output full dimensionality (stable for downstream Linear layers).
        total = self.freqs.numel()
        n_active = max(1, int(self.progress * total))
        outs = [x] if self.include_input else []
        for i, f in enumerate(self.freqs):
            mask = 1.0 if i < n_active else 0.0  # zero out inactive bands
            s = torch.sin(x * f) * mask
            c = torch.cos(x * f) * mask
            outs.append(s)
            outs.append(c)
        return torch.cat(outs, dim=-1)


class CoordField(nn.Module):
    """Coordinate MLP: 3D coords -> D feature vector"""
    def __init__(self, out_dim=32, hidden=256, depth=8, encoding=None):
        super().__init__()
        self.encoding = encoding
        in_dim = 3
        if encoding is not None:
            # note: encoding increases input dim dynamically at forward time,
            # so compute a rough estimate by encoding a dummy tensor
            with torch.no_grad():
                in_dim = encoding(torch.zeros(1, 3).cuda()).shape[-1]

        self.act = nn.ReLU(inplace=True)
        self.skip_at = max(1, depth // 2)  # insert skip halfway (like NeRF)
        self.layers = nn.ModuleList()
        for i in range(depth):
            in_ch = in_dim if i == 0 else hidden
            if i == self.skip_at:
                in_ch += in_dim  # concat skip input
            self.layers.append(nn.Linear(in_ch, hidden))
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, xyz):
        # xyz: (N, 3)
        if self.encoding is not None:
            xyz_enc = self.encoding(xyz)
        else:
            xyz_enc = xyz

        h = xyz_enc
        for i, layer in enumerate(self.layers):
            if i == self.skip_at:
                h = torch.cat([h, xyz_enc], dim=-1)
            h = self.act(layer(h))
        return self.head(h)  # (N, out_dim)


def median_nn_distance(xyz: torch.Tensor, sample_size=4096):
    N = xyz.shape[0]
    if N > sample_size:
        idx = torch.randperm(N, device=xyz.device)[:sample_size]
        sub = xyz[idx]
    else:
        sub = xyz
    dmat = torch.cdist(sub, sub)
    dmat += torch.eye(dmat.shape[0], device=xyz.device) * 1e6
    nn = dmat.min(dim=1).values
    return nn.median().item()


def train_on_scene(xyz_np: np.ndarray, feats_np: np.ndarray, format="emb", 
                   iters=20000, lr=1e-4, device="cuda"):
    # ...existing code before normalization...
    xyz = torch.from_numpy(xyz_np.astype(np.float32)).to(device)
    xyz_org = xyz.clone()
    mean = xyz.mean(0, keepdim=True)
    xyz_centered = xyz - mean

    # Raw median spacing (before any scaling)
    raw_med = median_nn_distance(xyz_centered)
    print(f"Raw median 1-NN distance: {raw_med:.6f}")

    # Spacing-aware scaling: choose target normalized median spacing
    target_med = 0.02  # adjust (0.01–0.05 reasonable)
    scale_factor = raw_med / target_med
    xyz_scaled = xyz_centered / (scale_factor + 1e-9)

    # Report normalized spacing
    norm_med = median_nn_distance(xyz_scaled)
    print(f"Normalized median 1-NN distance: {norm_med:.6f}")

    # Derive max_freq: π / d_norm (Nyquist-ish), then clamp
    max_freq = np.clip(np.pi / max(norm_med, 1e-6), 5.0, 60.0)
    num_bands = 16
    freqs = np.logspace(0.0, np.log10(max_freq), num_bands)
    freqs_t = torch.tensor(freqs, dtype=torch.float32, device=device)
    print(f"Using max_freq={max_freq:.2f} (num_bands={num_bands})")

    feats = torch.from_numpy(feats_np.astype(np.float32)).to(device)

    encoding = ProgressiveFourierEncoding(freqs_t, include_input=True).to(device)

    model = CoordField(out_dim=feats.shape[1], hidden=512, depth=8, encoding=encoding).to(device)
    converter = Converter("checkpoints_exp/checkpoint_sfvae.pth")

    opt = torch.optim.Adam(list(model.parameters()), lr=lr, weight_decay=0)
    mse = nn.MSELoss()
    lpips_loss = lpips.LPIPS(net='vgg').to(device)

    xyz_norm = xyz_scaled  # renamed for clarity

    for i in range(iters):
        # Progressive band activation first 20% iterations
        prog = min(1.0, (i + 1) / (0.2 * iters))
        encoding.set_progress(prog)

        opt.zero_grad()
        pred = model(xyz_norm)
        loss = mse(pred, feats)
        loss.backward()
        opt.step()

        if (i + 1) % 1000 == 0:
            active_bands = max(1, int(prog * num_bands))
            print(f"iter {i+1}/{iters}, loss={loss.item():.6f}, bands_active={active_bands}")

        if (i + 1) % 5000 == 0:
            print("running visual validation...")
            with torch.no_grad():
                pred_full = model(xyz_norm)
                if format == "emb":
                    pred_full = pred_full.squeeze(0)
                    bs = 1024
                    res, gt = [], []
                    for j in range(0, pred_full.shape[0], bs):
                        batch = pred_full[j:j+bs]
                        batch_gt = feats[j:j+bs]
                        gs = converter(batch, mode="emb2gaussian")
                        gt_gs = converter(batch_gt, mode="emb2gaussian")
                        res.append(gs); gt.append(gt_gs)
                    gt = torch.cat(gt, dim=0)
                    res = torch.cat(res, dim=0)
                    # restore original (unscaled) spatial coords for visualization
                    res[:, :3] = xyz_org - mean
                    gt[:, :3] = xyz_org - mean
                else:
                    res = torch.cat([xyz_org-mean, pred_full], dim=-1)
                    gt = torch.cat([xyz_org-mean, feats], dim=-1)

                pred_img = visualize_gaussian(res, camera_distance=5.0, h=1080, w=1080,
                                              save_img=True, filename=f"nf_{format}_iter{i+1}.png")
                gt_img = visualize_gaussian(gt, camera_distance=5.0, h=1080, w=1080,
                                            save_img=True, filename=f"nf_{format}_gt.png")
                psnr_value = psnr(torch.tensor(gt_img), torch.tensor(pred_img)).mean().item()
                ssim_value = ssim(gt_img, pred_img).mean().item()
                lpips_value = lpips_loss(torch.tensor(pred_img).permute(2,0,1).unsqueeze(0).to(device),
                                         torch.tensor(gt_img).permute(2,0,1).unsqueeze(0).to(device)).mean().item()
                print(f"Validation PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}, LPIPS: {lpips_value:.4f}")

    # Store scale_factor instead of previous max-abs scale
    state = {"model": model.state_dict(), "mean": mean.cpu(), "scale": torch.tensor(scale_factor).cpu(),
             "max_freq": max_freq, "num_bands": num_bands}
    return model, state


def collate_fn(batch):
    item = batch[0]
    xyz = item['xyz']
    feats = item['emb']
    return xyz, feats


def collate_fn_param(batch):
    item = batch[0]
    # print(item.keys())
    xyz = item[:, :3]
    param = item[:, 3:]
    return xyz, param


if __name__ == "__main__":

    from argparse import ArgumentParser
    from dataset import PlyObjectEmbedding, PlyObject

    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str)
    parser.add_argument("--format", type=str, default="param", choices=["emb", "param"])
    parser.add_argument("--iters", type=int, default=10000)
    args = parser.parse_args()

    if args.format == "emb":
        dataset = PlyObjectEmbedding(args.src_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    else:
        dataset = PlyObject(args.src_path, return_type="param")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_param)

    data = next(iter(dataloader))
    xyz_np = data[0]
    feats_np = data[1]
    model, state = train_on_scene(xyz_np, feats_np, format=args.format, iters=args.iters, lr=1e-3, device="cuda")
