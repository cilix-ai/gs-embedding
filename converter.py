from embedding_model import SFVAE, downsample_points
from dataset import PlyObject, PlyObjectEmbedding, Gaussian
from torch.utils.data import DataLoader
from utils.gs_utils import point2gaussian_torch_batched
from utils.visualize import visualize_gaussian
from utils.eval_metrics import *

import torch
import torch.nn as nn
import numpy as np
import tqdm
import os


class Converter(nn.Module):

    def __init__(self, checkpoint_path, embedding_dim=32, grid_dim=24, device=0):
        super(Converter, self).__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        self.embedding_model = SFVAE(embedding_dim=embedding_dim, grid_dim=grid_dim, deterministic=True)
        self.embedding_model = torch.compile(self.embedding_model)
        # if not compile, remove "_orig_mod" prefix
        # for key in list(checkpoint['state_dict'].keys()):
        #     if key.startswith("_orig_mod."):
        #         new_key = key[len("_orig_mod."):]
        #         checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(key)

        self.embedding_model.load_state_dict(checkpoint['state_dict'])
        self.embedding_model = self.embedding_model.cuda(device)
        self.embedding_model.eval()


    def forward(self, data, mode):
        with torch.no_grad():
            if mode == "gaussian2emb":
                return self.embedding_model.encode(data)
            elif mode == "emb2gaussian":
                pred = self.embedding_model.decode(data)
                param = point2gaussian_torch_batched(pred)
                return param
            else:
                raise ValueError("Invalid mode")


def gaussian2emb(converter, src_path, dist_pth, device=0):
    os.makedirs(dist_pth, exist_ok=True)
    dataset = PlyObject(path=src_path, num_points=12*12)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, data in enumerate(dataloader):
        _, obj_points, _ = data['g_centroids'], data['obj_points'], data['cls']
        obj_points = torch.tensor(obj_points).float().cuda(device)

        obj_embedding = []

        inner_bs = 1024
        with torch.no_grad():
            num_points = obj_points.shape[1]
            bar = tqdm.tqdm(total=num_points)
            for j in range(0, num_points, inner_bs):
                end_idx = min(j + inner_bs, num_points)
                batched_points = obj_points[:, j:end_idx]
                batched_points = batched_points.view(-1, batched_points.shape[2], batched_points.shape[3])
                embedding = converter(batched_points, mode="gaussian2emb")
                obj_embedding.extend(embedding.cpu().numpy())
                actual_update = end_idx - j
                bar.update(actual_update)
            bar.close()

        # save as npz
        tokenized = {}
        tokenized['xyz'] = np.array(data['g_centroids']).squeeze(axis=0)
        tokenized['emb'] = np.stack(obj_embedding)
        path = f"{dist_pth}/{i}.npz"
        print(f"Saving data to {path}")
        np.savez(path, **tokenized)


def emb2gaussian_batched(converter, src_path, dist_pth, device=0):
    dataset = PlyObjectEmbedding(src_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, data in enumerate(dataloader):
        output_est = []
        g_centroids, obj_embeddings, cls = data['xyz'], data['emb'], data['cls']
        obj_embeddings = torch.tensor(obj_embeddings).float().cuda(device)
        g_centroids = g_centroids.squeeze(0).to(device)

        with torch.no_grad():
            num_gaussians = obj_embeddings.shape[1]
            batch_size = 4096
            bar = tqdm.tqdm(total=num_gaussians)
            for i in range(0, num_gaussians, batch_size):
                end_idx = min(i + batch_size, num_gaussians)
                batched_embeddings = obj_embeddings[:, i:end_idx]
                batched_embeddings = batched_embeddings.squeeze(0)  # (1, N, 32)
                est_gaussians = converter(batched_embeddings, mode="emb2gaussian")
                output_est.extend(est_gaussians)
                bar.update(end_idx - i)
            bar.close()
    
        output_gs = []
        for l in output_est:
            output_gs.append(Gaussian.list2gaussian(l))
        
        dataset._save_ply(dist_pth, output_gs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian2emb", action="store_true")
    parser.add_argument("--emb2gaussian", action="store_true")
    parser.add_argument("--src_path", type=str, default=None, required=True)
    parser.add_argument("--dist_path", type=str, default=None, required=True)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    converter = Converter("checkpoints_exp/checkpoint_sfvae_sh0_144.pth", 
                          embedding_dim=32, grid_dim=12, device=args.cuda)

    if args.gaussian2emb:
        gaussian2emb(
            converter, 
            args.src_path,
            args.dist_path,
            device=args.cuda
        )
    elif args.emb2gaussian:
        emb2gaussian_batched(
            converter, 
            args.src_path,
            args.dist_path,
            device=args.cuda
        )
    else:
        print("Please specify conversion mode")
