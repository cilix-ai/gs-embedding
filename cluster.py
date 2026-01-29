from __future__ import annotations

import argparse
import math
import pathlib
import random
from typing import Tuple

import numpy as np
from tqdm import tqdm

# Optional deps handled gracefully
try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None  # type: ignore

try:
    import hdbscan  # type: ignore
except ImportError:
    hdbscan = None

# Optional Leiden support
try:
    import igraph as ig  # type: ignore
    import leidenalg  # type: ignore
except ImportError:
    ig = None  # type: ignore
    leidenalg = None  # type: ignore

import matplotlib.pyplot as plt


def load_scene(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return means (N,3), latent (N,D) and colours (N,3 or None)."""
    data = np.load(path)
    xyz = data["xyz"].astype(np.float32)
    emb = data["emb"].astype(np.float32) if "emb" in data else None
    rgb = data["rgb"].astype(np.float32) if "rgb" in data else None
    return xyz, emb, rgb


def normalise_features(means: np.ndarray, latent: np.ndarray | None, *,
                       spatial_weight: float) -> np.ndarray:
    """Concatenate (weighted) xyz and latent into a single feature matrix."""
    # Convert metres→centimetres so spatial and latent magnitudes are comparable
    means_cm = means * 100.0 * spatial_weight
    if latent is None:
        return means_cm

    # Whitening latent improves k‑means convergence
    latent_std = latent.std(axis=0, keepdims=True) + 1e-9
    latent_norm = latent / latent_std
    return np.hstack([means_cm, latent_norm])


def choose_distinct_colours(n: int) -> np.ndarray:
    """Return an (n,3) uint8 array of visually distinct colours."""
    cmap = plt.get_cmap('tab20')
    if n <= cmap.N:
        colors = (np.array([cmap(i)[:3] for i in range(n)]) * 255).astype(np.uint8)
    else:
        rng = np.random.default_rng(42)
        colors = (rng.random((n, 3)) * 255).astype(np.uint8)
    return colors


def write_ply(path: pathlib.Path, points: np.ndarray, colors: np.ndarray):
    """Write ASCII PLY with x,y,z,r,g,b."""
    assert points.shape[0] == colors.shape[0]
    # recenter points
    points = points - points.mean(axis=0, keepdims=True)
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def cluster_kmeans(features: np.ndarray, k: int, seed: int) -> np.ndarray:
    if KMeans is None:
        raise ImportError("scikit‑learn required for k‑means; pip install scikit‑learn")
    model = KMeans(n_clusters=k, random_state=seed, n_init='auto', verbose=0)
    labels = model.fit_predict(features)
    return labels.astype(np.int32)


def cluster_hdbscan(features: np.ndarray, min_cluster_size: int,
                    min_samples: int | None, eps: float | None) -> np.ndarray:
    if hdbscan is None:
        raise ImportError("hdbscan package not found; pip install hdbscan")
    kwargs = dict(min_cluster_size=min_cluster_size)
    if min_samples is not None:
        kwargs['min_samples'] = min_samples
    if eps is not None:
        kwargs['cluster_selection_epsilon'] = eps
    clusterer = hdbscan.HDBSCAN(**kwargs)
    labels = clusterer.fit_predict(features)
    return labels.astype(np.int32)


def cluster_leiden(features: np.ndarray, k: int, resolution: float,
                   seed: int) -> np.ndarray:
    if ig is None or leidenalg is None:
        raise ImportError("igraph + leidenalg required for Leiden; pip install igraph leidenalg")
    # Build k‑NN graph (cosine similarity) – use a naive approach for ≤500 k points
    from sklearn.neighbors import NearestNeighbors  # local import to avoid heavy dep if unused
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(features)
    knn_idx = nbrs.kneighbors(return_distance=False)

    g = ig.Graph(n=len(features), directed=False)
    # Add edges
    edges = [(i, j) for i, row in enumerate(knn_idx) for j in row if i < j]
    g.add_edges(edges)
    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                         resolution_parameter=resolution, seed=seed)
    labels = np.array(partition.membership, dtype=np.int32)
    return labels


def cluster_compactness(features: np.ndarray, labels: np.ndarray) -> Tuple[dict, float]:
    compactness = {}
    total_weighted = 0.0
    total_points = 0

    # Process only non-noise clusters
    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return {}, float('nan')

    for cid in np.unique(labels[valid_mask]):
        idx = np.where(labels == cid)[0]
        pts = features[idx]
        if len(pts) == 0:
            continue
        centroid = pts.mean(axis=0, keepdims=True)
        mse = float(np.mean(np.sum((pts - centroid) ** 2, axis=1)))
        compactness[int(cid)] = mse
        total_weighted += mse * len(pts)
        total_points += len(pts)

    avg_compactness = total_weighted / total_points if total_points > 0 else float('nan')
    return compactness, avg_compactness


def silhouette_coefficient(features: np.ndarray, labels: np.ndarray) -> float:
    try:
        from sklearn.metrics import silhouette_score  # type: ignore
    except ImportError:
        raise ImportError("scikit-learn required for silhouette_score; pip install scikit‑learn")

    mask = labels >= 0
    if np.sum(mask) == 0:
        return float('nan')

    lbls = labels[mask]
    feats = features[mask]

    # Need at least 2 distinct clusters for silhouette
    if len(np.unique(lbls)) < 2:
        return float('nan')

    return float(silhouette_score(feats, lbls, metric='euclidean'))


def stratified_downsample(labels: np.ndarray, sample_size: int, seed: int = 0) -> np.ndarray:
    """
    Return indices of a stratified random subset of points, preserving the
    cluster distribution. Noise (-1) is included proportionally.
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    if sample_size >= n:
        return np.arange(n)

    # Compute target counts per label
    unique, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    target_counts = np.floor(proportions * sample_size).astype(int)

    # Ensure we hit the total exactly
    deficit = sample_size - target_counts.sum()
    if deficit > 0:
        # Distribute remaining slots to largest clusters first
        order = np.argsort(-counts)
        for i in order[:deficit]:
            target_counts[i] += 1

    # Collect indices per label
    idxs = []
    for lbl, tcnt in zip(unique, target_counts):
        if tcnt == 0:
            continue
        pool = np.where(labels == lbl)[0]
        # Sample without replacement
        sel = rng.choice(pool, size=tcnt, replace=False)
        idxs.append(sel)

    return np.concatenate(idxs, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Unsupervised clustering of Gaussians with quick colouring output.")
    parser.add_argument("--input", type=str, required=True, help="Path to .npz scene with means + latent (+ rgb)")
    parser.add_argument("--output", type=str, required=True, help="Output PLY or NPZ path")

    parser.add_argument("--method", choices=["kmeans", "hdbscan", "leiden"], default="kmeans")
    parser.add_argument("--k", type=int, default=30, help="Number of clusters for k‑means or neighbour count for Leiden")
    parser.add_argument("--min-cluster-size", type=int, default=300, help="HDBSCAN: minimum cluster size")
    parser.add_argument("--min-samples", type=int, default=None, help="HDBSCAN: min_samples")
    parser.add_argument("--eps", type=float, default=None, help="HDBSCAN: cluster_selection_epsilon")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution parameter")

    parser.add_argument("--spatial-weight", type=float, default=0.25, help="Multiply xyz (in cm) by this weight in the feature vector")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    in_path = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output)

    print(f"[+] Loading scene {in_path} …")
    means, latent, rgb = load_scene(in_path)
    print(f"    {len(means):,} Gaussians, latent dim = {None if latent is None else latent.shape[1]}")

    print("[+] Building feature matrix …")
    features = normalise_features(means, latent, spatial_weight=args.spatial_weight)

    print(f"[+] Clustering using {args.method} …")
    if args.method == "kmeans":
        labels = cluster_kmeans(features, args.k, args.seed)
    elif args.method == "hdbscan":
        labels = cluster_hdbscan(features, args.min_cluster_size, args.min_samples, args.eps)
    else:  # Leiden
        labels = cluster_leiden(features, args.k, args.resolution, args.seed)

    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    print(f"    Found {n_clusters} clusters (+ noise)")

    # Compute metrics
    subset_size = min(20_000, len(labels))
    sub_idx = stratified_downsample(labels, subset_size, seed=args.seed)
    feats_sub = features[sub_idx]
    labels_sub = labels[sub_idx]

    sil_coeff = silhouette_coefficient(feats_sub, labels_sub)
    print(f"    Silhouette coefficient (subset {len(sub_idx)}): {sil_coeff:.6f}")
    comp_per_cluster, comp_avg = cluster_compactness(feats_sub, labels_sub)
    print(f"    Average cluster compactness (MSE, subset): {comp_avg:.6f}")

    # Assign colors
    cluster_colors = choose_distinct_colours(max(n_clusters, 1))
    colors_vis = np.zeros((len(labels), 3), dtype=np.uint8)
    for idx, cid in enumerate(labels):
        colors_vis[idx] = cluster_colors[cid] if cid >= 0 else [128, 128, 128]  # grey for noise

    # Save
    if out_path.suffix.lower() == ".ply":
        print(f"[+] Writing coloured PLY to {out_path} …")
        write_ply(out_path, means, colors_vis)
    else:
        print(f"[+] Writing NPZ to {out_path} …")
        np.savez_compressed(out_path, means=means, latent=latent, rgb=rgb,
                            cluster_id=labels, rgb_vis=colors_vis)

    print("[✓] Done.")


if __name__ == "__main__":
    main()
