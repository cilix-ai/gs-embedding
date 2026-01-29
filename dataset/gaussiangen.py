import numpy as np
# from scipy.spatial.transform import Rotation as Rot
from torch.utils.data import Dataset
from plyfile import PlyData, PlyElement
from utils.gs_utils import *


class Gaussian:

    def __init__(self, 
                 centroid:np.ndarray,
                 scale:np.ndarray, 
                 rotation:np.ndarray, 
                 opacity:float, 
                 feat_dc:np.ndarray,
                 feat_extra:np.ndarray):

        assert centroid.shape == (3,)
        assert scale.shape == (3,)
        assert rotation.shape == (4,)
        assert feat_dc.shape == (3,)

        self.centroid = centroid
        self.scale = scale
        self.rotation = rotation
        self.opacity = opacity
        self.max_sh_degree = 3
        self.feat_dc = feat_dc
        self.feat_extra = feat_extra


    def to_dict(self):
        return {
            'centroid': self.centroid,
            'scale': self.scale,
            'rotation': self.rotation,
            'opacity': self.opacity,
            'feat_dc': self.feat_dc,
            'feat_extra': self.feat_extra
        }
    

    def to_list(self):
        return np.concatenate(
            [self.centroid.flatten(), self.scale.flatten(), 
             self.rotation.flatten(), self.opacity.flatten(), 
             self.feat_dc.flatten(), self.feat_extra.flatten()]
        )
    

    @staticmethod
    def list2gaussian(data):
        assert len(data.shape) == 1
        assert data.shape[0] >= 3 + 3 + 4 + 1 + 3  # centroid + scale + rotation + opacity + feat_dc

        centroid = data[0:3]
        scale = data[3:6]
        rotation = data[6:10]
        opacity = data[10]
        feat_dc = data[11:14]
        feat_extra = data[14:]

        g = Gaussian(centroid=centroid, scale=scale, rotation=rotation,
                     opacity=opacity, feat_dc=feat_dc, feat_extra=feat_extra)
        return g


    def activate(self):
        # exponent activation for scale
        scale = np.exp(self.scale)
        # normalization for rotation
        rotation = self.rotation / np.linalg.norm(self.rotation)
        # sigmoid activation for opacity
        opacity = 1 / (1 + np.exp(-self.opacity))
        return scale, rotation, opacity


    def save_ply(self, path):
        """
        !!! A test implementation
        """
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self.feat_dc.shape[0]):
            l.append(f'f_dc_{i}')
        for i in range(self.feat_extra.shape[0]):
            l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(self.scale.shape[0]):
            l.append(f'scale_{i}')
        for i in range(self.rotation.shape[0]):
            l.append(f'rot_{i}')

        dtype_full = [(attr, 'f4') for attr in l]

        xyz = np.array([.0, .0, .0])
        normals = np.array([.0, .0, .0])

        elements = np.empty(1, dtype=dtype_full)
        attrs = np.concatenate((xyz, normals, self.feat_dc, 
                                self.feat_extra.flatten(), [self.opacity], 
                                self.scale, self.rotation), axis=0)
        elements[0] = tuple(attrs)
        # elements[:] = list(map(tuple, attrs))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def gaussian2point(self, num_points=144, clip_color=False):
        # activate the parameters
        scale, rotation, opacity = self.activate()

        # R = Rot.from_quat(self.rotation).as_matrix()
        R = qvec2rotmat(rotation)

        extra_len = int(self.feat_extra.shape[0]/3)
        coeff_r = np.concatenate(([self.feat_dc[0]], self.feat_extra[:extra_len]))
        coeff_g = np.concatenate(([self.feat_dc[1]], self.feat_extra[extra_len:2*extra_len]))
        coeff_b = np.concatenate(([self.feat_dc[2]], self.feat_extra[2*extra_len:]))
        sh = np.array([coeff_r, coeff_g, coeff_b])

        size = np.sqrt(num_points).astype(int)
        u_ = np.linspace(0, 2 * np.pi, size)
        v_ = np.linspace(0, np.pi, size)
        u, v = np.meshgrid(u_, v_)

        x, y, z, color = self._uv_to_xyz_color(u, v, scale, R, sh)
        if clip_color:
            color = np.clip(color + 0.5, a_min=0.0, a_max=1.0)

        color = color.reshape(-1, 3)
        x, y, z = x.flatten(), y.flatten(), z.flatten()
        opacity = np.repeat(opacity, num_points) # repeat opacity for each point

        points = np.concatenate([x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis],
                                 color, opacity[:, np.newaxis]], axis=-1)

        return points


    def _uv_to_xyz_color(self, u, v, scale, R, sh):
        x = np.sin(v) * np.cos(u)
        y = np.sin(v) * np.sin(u)
        z = np.cos(v)
        scale_x = scale[0]
        scale_y = scale[1]
        scale_z = scale[2]

        points = np.array([scale_x * x.flatten(), scale_y * y.flatten(), scale_z *z.flatten()])
        rotated_points = R @ points
        x_rot = rotated_points[0, :].reshape(x.shape)
        y_rot = rotated_points[1, :].reshape(y.shape)
        z_rot = rotated_points[2, :].reshape(z.shape)

        dirs = np.array([x.ravel()[np.newaxis, :],
                         y.ravel()[np.newaxis, :],
                         z.ravel()[np.newaxis, :]]).T

        unnomralized_colors = eval_sh(self.max_sh_degree, sh, dirs)

        return x_rot, y_rot, z_rot, unnomralized_colors


    @staticmethod
    def point2gaussian(data):
        assert data.shape[1] == 7

        points, colors, opacities = data[:, :3], data[:, 3:6], data[:, -1]
        abc, R, _ = fit_ellipsoid_pca(points) 
        centroid = np.array([0, 0, 0])
        xyz = ellipsoid_xyz2dirs(points, centroid, qvec2rotmat(R), abc)
        sh = fit_sh(xyz, colors, deg=3)

        abc = np.array(abc)
        sh = sh.T
        feat_dc = sh[:, 0]
        feat_extra = sh[:, 1:].flatten()
        scale = np.log(abc)
        opacity = sigmoid_inverse(np.mean(opacities))

        g = Gaussian(centroid=centroid, rotation=R, scale=scale,
                     opacity=opacity, feat_dc=feat_dc, feat_extra=feat_extra)
        return g


class GaussianGen(Dataset):

    def __init__(self, num_samples=1000, num_points=12*12, 
                 max_scale=-2., min_scale=-12.,
                 max_opacity=10, min_opacity=-5,
                 sh_degree=3, return_type="gaussian"):
        self.num_samples = num_samples
        self.num_points = num_points
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.max_opacity = max_opacity
        self.min_opacity = min_opacity
        self.max_sh_degree = 3
        self.sh_degree = sh_degree

        assert return_type in ["gaussian", "param", "both"], \
            f"Invalid return_type: {return_type}"
        self.return_type = return_type

        print(f"Generating {num_samples} Gaussians...")
        self.gaussians = []
        for _ in range(num_samples):
            self.gaussians.append(self.generate())

        # normalize the parameters if return_type is "param"
        # if self.return_type == "param":
        #     self.gaussians = np.array([g.to_list() for g in self.gaussians])
        #     mean, max, min = np.mean(self.gaussians, axis=0), \
        #                      np.max(self.gaussians, axis=0), \
        #                      np.min(self.gaussians, axis=0)
        #     self.gaussians = normalize_gaussian_param(self.gaussians, mean, max, min)


    def __getitem__(self, index):
        g = self.gaussians[index]
        if self.return_type == "gaussian":
            return g.gaussian2point(self.num_points)
        elif self.return_type == "param":
            return g.to_list()
        else:
            return g.gaussian2point(self.num_points), g.to_list()


    def __len__(self):
        return self.num_samples


    def generate(self):
        # random scale
        scale_x = np.random.uniform(self.min_scale, self.max_scale)
        scale_y = np.random.uniform(self.min_scale, self.max_scale)
        scale_z = np.random.uniform(self.min_scale, self.max_scale)

        # random rotation
        R = np.linalg.qr(np.random.randn(3, 3))[0]

        # random opacity
        opacity = np.array(np.random.uniform(self.min_opacity, self.max_opacity))

        # random spherical harmonics coefficients
        coeff_r = self._random_sh_coeff()
        coeff_g = self._random_sh_coeff()
        coeff_b = self._random_sh_coeff()

        # build Gaussian
        centroid = np.array([0.0, 0.0, 0.0])
        scale = np.array([scale_x, scale_y, scale_z])
        # rotation = Rot.from_matrix(R).as_quat()
        rotation = rotmat2qvec(R)
        feat_dc = np.array([coeff_r[0], coeff_g[0], coeff_b[0]])
        feat_extra = np.concatenate([coeff_r[1:], coeff_g[1:], coeff_b[1:]])

        return Gaussian(centroid, scale, rotation, opacity, feat_dc, feat_extra)


    @staticmethod
    def gaussian_param_interpolate(self, idx_1, idx_2, steps=5, num_points=12*12):
        A = self.gaussians[idx_1]
        B = self.gaussians[idx_2]

        interpolates = []

        for i in range(steps + 1):
            alpha = i / steps
            interpolate = lambda a, b: alpha * a + (1 - alpha) * b
            inter_centroid = np.array([0.0, 0.0, 0.0])
            inter_scale = interpolate(A.scale, B.scale)
            inter_rotation = interpolate(A.rotation, B.rotation)
            inter_opacity = interpolate(A.opacity, B.opacity)
            inter_feat_dc = interpolate(A.feat_dc, B.feat_dc)
            inter_feat_extra = interpolate(A.feat_extra, B.feat_extra)
            inter_gaussian = Gaussian(inter_centroid, inter_scale, inter_rotation, 
                                      inter_opacity, inter_feat_dc, inter_feat_extra)

            # points = inter_gaussian.gaussian2point(num_points)
            # interpolates.append(points)
            interpolates.append(inter_gaussian)

        return interpolates


    def _random_sh_coeff(self):
        # Exponentially decaying standard deviation for higher degrees
        # variances = [1 / (1 + d) for d in range(degree + 1)]
        degree = self.sh_degree
        variance_decay = 4
        variances = [1 / (variance_decay ** d) for d in range(degree + 1)]
        coeff = []

        # Generate coefficients band by band
        for d in range(degree + 1):
            var = variances[d]
            band_coeff = np.random.normal(0, var, (2 * d + 1))  # Coeffs for this degree
            coeff.extend(band_coeff)

        if degree < self.max_sh_degree:
            void_space = (self.max_sh_degree + 1) ** 2 - (degree + 1) ** 2
            random_noise = np.random.normal(0, 0.05, void_space)
            coeff.extend(random_noise)

        # Normalize coefficients to have bounded energy
        coeff = np.array(coeff)
        # coeff /= np.linalg.norm(coeff) + 1e-8  # Normalize to unit norm (optional scaling can be added)

        return coeff


if __name__ == '__main__':
    gg = GaussianGen()
    gaussian = next(iter(gg))
    print(gaussian.shape)
