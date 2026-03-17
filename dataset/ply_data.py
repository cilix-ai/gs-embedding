from plyfile import PlyData, PlyElement
import numpy as np
from torch.utils.data import Dataset
from dataset.gaussiangen import Gaussian
import glob
import os
import tqdm
from collections import OrderedDict
import torch
from utils.gs_utils import gaussian2point_torch_batch


class PlyBase(Dataset):
    """
    Ply dataset in 3D Gaussian splatting format.
    Implementation based on 'https://github.com/qimaqi/ShapeSplat-Gaussian_MAE'
    """

    def __init__(self):
        super(PlyBase, self).__init__()


    def _parse_ply(self, path):
        gaussians = []

        gs_vertex = PlyData.read(path)['vertex']

        # load centroids[x,y,z] - Gaussian centroid
        x = gs_vertex['x'].astype(np.float32)
        y = gs_vertex['y'].astype(np.float32)
        z = gs_vertex['z'].astype(np.float32)
        centroids = np.stack((x, y, z), axis=-1) # [n, 3]

        # load o - opacity
        opacity = gs_vertex['opacity'].astype(np.float32).reshape(-1, 1)

        # load scales[sx, sy, sz] - Scale
        scale_names = [
            p.name
            for p in gs_vertex.properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((centroids.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = gs_vertex[attr_name].astype(np.float32)

        # load rotation rots[q_0, q_1, q_2, q_3] - Rotation
        rot_names = [
            p.name for p in gs_vertex.properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((centroids.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = gs_vertex[attr_name].astype(np.float32)

        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)

        # load base sh_base[dc_0, dc_1, dc_2] - Spherical harmonic
        sh_base = np.zeros((centroids.shape[0], 3, 1))
        sh_base[:, 0, 0] = gs_vertex['f_dc_0'].astype(np.float32)
        sh_base[:, 1, 0] = gs_vertex['f_dc_1'].astype(np.float32)
        sh_base[:, 2, 0] = gs_vertex['f_dc_2'].astype(np.float32)
        sh_base = sh_base.reshape(-1, 3)

        # load extra sh_extra[rest_0, rest_1, ..., rest_44] - Extra features
        sh_extra = np.zeros((centroids.shape[0], 45, 1))
        for i in range(45):
            sh_extra[:, i, 0] = gs_vertex[f'f_rest_{i}'].astype(np.float32)

        for i in range(centroids.shape[0]):
            gaussians.append(
                Gaussian(
                    centroid=centroids[i],
                    scale=scales[i],
                    rotation=rots[i],
                    opacity=opacity[i],
                    feat_dc=sh_base[i],
                    feat_extra=sh_extra[i].flatten()
                )
            )

        return gaussians


    def _save_ply(self, path, gaussians):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(gaussians[0].feat_dc.shape[0]):
            l.append(f'f_dc_{i}')
        for i in range(gaussians[0].feat_extra.shape[0]):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(gaussians[0].scale.shape[0]):
            l.append(f'scale_{i}')
        for i in range(gaussians[0].rotation.shape[0]):
            l.append(f'rot_{i}')

        dtype_full = [(attr, 'f4') for attr in l]

        data = np.zeros((len(gaussians),), dtype=dtype_full)

        for i, g in enumerate(gaussians):
            data[i]['x'] = g.centroid[0]
            data[i]['y'] = g.centroid[1]
            data[i]['z'] = g.centroid[2]
            data[i]['nx'] = 0
            data[i]['ny'] = 0
            data[i]['nz'] = 0
            for j in range(g.feat_dc.shape[0]):
                data[i][f'f_dc_{j}'] = g.feat_dc[j]
            for j in range(g.feat_extra.shape[0]):
                data[i][f'f_rest_{j}'] = g.feat_extra[j]
            data[i]['opacity'] = g.opacity
            for j in range(g.scale.shape[0]):
                data[i][f'scale_{j}'] = g.scale[j]
            for j in range(g.rotation.shape[0]):
                data[i][f'rot_{j}'] = g.rotation[j]

        el = PlyElement.describe(data, 'vertex')
        ply_data = PlyData([el])
        ply_data.write(path)


class Ply(PlyBase):
    """Use for embedding model training"""

    def __init__(self, path, num_points=144, random_choose=0, return_type="gaussian"):
        super(Ply, self).__init__()
        self.path = path
        self.num_points = num_points

        assert return_type in ["gaussian", "param", "both"], \
            "return_type must be 'gaussian', 'param' or 'both'"
        self.return_type = return_type
        self.gaussians = self._load_ply(path, random_choose)


    def __len__(self):
        return len(self.gaussians)


    def __getitem__(self, idx):
        if self.return_type == "gaussian":
            g = self.gaussians[idx]
            return g.gaussian2point(self.num_points)
        elif self.return_type == "param":
            return self.gaussians[idx].to_list()
        else:  # both
            g = self.gaussians[idx]
            return g.gaussian2point(self.num_points), g.to_list()


    def _load_ply(self, path, random_choose=0):
        print(f"Loading data from {path}")
        # get all ply files recursively
        files = glob.glob(os.path.join(path, '**', '*.ply'), recursive=True)
        if random_choose:
            files_chosen = int(len(files) * random_choose)
            files = np.random.choice(files, files_chosen, replace=False)

        # store data as separate Gaussians
        gaussians = []

        for f in files:
            gaussians.extend(self._parse_ply(f))

        return gaussians


class PlyObject(PlyBase):
    """Object-level dataset with LRU caching"""

    def __init__(self, path, num_points=144, cache_size=50, random_choose=0, return_type="gaussian"):
        super(PlyObject, self).__init__()
        self.path = path
        self.num_points = num_points
        self.files = self._get_files(path, random_choose)
        self.cache_size = cache_size
        assert return_type in ["gaussian", "param", "both"], \
            "return_type must be 'gaussian', 'param' or 'both'"
        self.return_type = return_type

        self.gaussian_objs = OrderedDict()  # LRU cache
        self._prefill_cache()


    def _get_files(self, path, random_choose):
        print(f"Loading file paths from {path}")
        # get all ply files recursively
        files = glob.glob(os.path.join(path, '**', '*.ply'), recursive=True)
        if random_choose:
            files_chosen = int(len(files) * random_choose)
            files = np.random.choice(files, files_chosen, replace=False)
        return files


    def _prefill_cache(self):
        print(f"Pre-filling cache with {self.cache_size} files")
        for i in range(min(self.cache_size, len(self.files))):
            file_path = self.files[i]
            self.gaussian_objs[file_path] = self._load_ply(file_path)


    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        file_path = self.files[idx]
        if file_path not in self.gaussian_objs:
            if len(self.gaussian_objs) >= self.cache_size:
                self.gaussian_objs.popitem(last=False)  # Remove the oldest item
            self.gaussian_objs[file_path] = self._load_ply(file_path)
        else:
            # Move the accessed item to the end to mark it as recently used
            self.gaussian_objs.move_to_end(file_path)
        g_obj, cls = self.gaussian_objs[file_path]

        if self.return_type == "param":
            g_params = np.array([g.to_list() for g in g_obj])
            return g_params

        # batched gpu processing
        obj_points = gaussian2point_torch_batch(g_obj, self.num_points, batch_size=1024, device='cpu')
        g_centroids = torch.stack([torch.tensor(g.centroid) for g in g_obj])

        if self.return_type == "both":
            g_params = np.array([g.to_list() for g in g_obj])
            return {
                'g_centroids': g_centroids,
                'obj_points': obj_points,
                'cls': cls
            }, g_params
        else:
            return {
                'g_centroids': g_centroids,
                'obj_points': obj_points,
                'cls': cls,
            }


    def _load_ply(self, file_path):
        gaussians = self._parse_ply(file_path)
        obj_class = os.path.basename(file_path)[:8]
        return gaussians, obj_class


class PlyObjectEmbedding(PlyBase):
    """Preprocessed object-level dataset with embeddings"""

    def __init__(self, path, random_choose=0, onehot_cls=False):
        self.path = path
        self.random_choose = random_choose
        self.onehot_cls = onehot_cls
        self.gaussian_objs = self._load_gaussian_obj(path)


    def _load_gaussian_obj(self, path):
        print(f"Loading data from {path}")
        # get all npz files recursively
        files = glob.glob(os.path.join(path, '**', '*.npz'), recursive=True)
        if self.random_choose:
            files_chosen = int(len(files) * self.random_choose)
            files = np.random.choice(files, files_chosen, replace=False)

        gaussian_objs = []
        unique_classes = set()

        bar = tqdm.tqdm(total=len(files))
        for f in files:
            data = np.load(f, allow_pickle=True)
            cls = os.path.basename(f).split('_')[0]
            unique_classes.add(cls)
            xyz = data['xyz']
            emb = data['emb']
            gaussian_objs.append({
                'xyz': xyz,
                'emb': emb, 
                'cls': cls, 
            })
            bar.update(1)
        bar.close()

        if self.onehot_cls:
            for g in gaussian_objs:
                # do onehot encoding for class
                onehot = np.zeros(len(unique_classes))
                onehot[list(unique_classes).index(g['cls'])] = 1
                g['cls'] = onehot

        return gaussian_objs


    def __len__(self):
        return len(self.gaussian_objs)


    def __getitem__(self, idx):
        return self.gaussian_objs[idx]
