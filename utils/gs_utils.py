import numpy as np
import torch


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def eval_sh_batch(deg, sh_batch, dirs_batch):
    """
    Evaluate spherical harmonics for a batch of SH coefficients and directions.
    
    Args:
        deg: int SH degree. Currently, 0-4 supported.
        sh_batch: torch.Tensor of shape (batch_size, C, (deg + 1) ** 2)
                  SH coefficients for each batch, where C is number of channels (e.g., 3 for RGB).
        dirs_batch: torch.Tensor of shape (batch_size, N, 3)
                    Unit directions for each batch, where N is number of directions.
    
    Returns:
        torch.Tensor of shape (batch_size, C, N)
        Evaluated SH values for each batch.
    """
    assert deg <= 4 and deg >= 0, "Only degrees 0-4 are supported."
    coeff = (deg + 1) ** 2
    assert sh_batch.shape[-1] >= coeff, f"SH coefficients shape {sh_batch.shape} insufficient for degree {deg}"
    assert dirs_batch.shape[-1] == 3, "Directions must have 3 components (x, y, z)."

    batch_size, num_channels, _ = sh_batch.shape
    _, num_dirs, _ = dirs_batch.shape

    # Extract directions: shape (batch_size, N, 1)
    x = dirs_batch[..., 0:1]  # (batch_size, N, 1)
    y = dirs_batch[..., 1:2]  # (batch_size, N, 1)
    z = dirs_batch[..., 2:3]  # (batch_size, N, 1)

    # Reshape sh_batch for broadcasting: (batch_size, C, 1, coeff)
    sh_batch = sh_batch.unsqueeze(2)  # (batch_size, C, 1, coeff)

    # Initialize result with degree 0: C0 * sh[..., 0]
    # sh_batch[..., 0] has shape (batch_size, C, 1), result shape (batch_size, C, N)
    result = C0 * sh_batch[..., 0].expand(-1, -1, num_dirs)

    if deg > 0:
        # Degree 1 terms
        # Each term: (batch_size, N, 1) * (batch_size, C, 1) -> (batch_size, C, N)
        result = (result -
                  C1 * (y.transpose(1, 2) * sh_batch[..., 1]).squeeze(2) +
                  C1 * (z.transpose(1, 2) * sh_batch[..., 2]).squeeze(2) -
                  C1 * (x.transpose(1, 2) * sh_batch[..., 3]).squeeze(2))

        if deg > 1:
            # Degree 2 terms
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            
            result = (result +
                      C2[0] * (xy.transpose(1, 2) * sh_batch[..., 4]).squeeze(2) +
                      C2[1] * (yz.transpose(1, 2) * sh_batch[..., 5]).squeeze(2) +
                      C2[2] * ((2.0 * zz - xx - yy).transpose(1, 2) * sh_batch[..., 6]).squeeze(2) +
                      C2[3] * (xz.transpose(1, 2) * sh_batch[..., 7]).squeeze(2) +
                      C2[4] * ((xx - yy).transpose(1, 2) * sh_batch[..., 8]).squeeze(2))

            if deg > 2:
                # Degree 3 terms
                result = (result +
                          C3[0] * ((y * (3 * xx - yy)).transpose(1, 2) * sh_batch[..., 9]).squeeze(2) +
                          C3[1] * ((xy * z).transpose(1, 2) * sh_batch[..., 10]).squeeze(2) +
                          C3[2] * ((y * (4 * zz - xx - yy)).transpose(1, 2) * sh_batch[..., 11]).squeeze(2) +
                          C3[3] * ((z * (2 * zz - 3 * xx - 3 * yy)).transpose(1, 2) * sh_batch[..., 12]).squeeze(2) +
                          C3[4] * ((x * (4 * zz - xx - yy)).transpose(1, 2) * sh_batch[..., 13]).squeeze(2) +
                          C3[5] * ((z * (xx - yy)).transpose(1, 2) * sh_batch[..., 14]).squeeze(2) +
                          C3[6] * ((x * (xx - 3 * yy)).transpose(1, 2) * sh_batch[..., 15]).squeeze(2))

                if deg > 3:
                    # Degree 4 terms
                    result = (result +
                              C4[0] * ((xy * (xx - yy)).transpose(1, 2) * sh_batch[..., 16]).squeeze(2) +
                              C4[1] * ((yz * (3 * xx - yy)).transpose(1, 2) * sh_batch[..., 17]).squeeze(2) +
                              C4[2] * ((xy * (7 * zz - 1)).transpose(1, 2) * sh_batch[..., 18]).squeeze(2) +
                              C4[3] * ((yz * (7 * zz - 3)).transpose(1, 2) * sh_batch[..., 19]).squeeze(2) +
                              C4[4] * ((zz * (35 * zz - 30) + 3).transpose(1, 2) * sh_batch[..., 20]).squeeze(2) +
                              C4[5] * ((xz * (7 * zz - 3)).transpose(1, 2) * sh_batch[..., 21]).squeeze(2) +
                              C4[6] * (((xx - yy) * (7 * zz - 1)).transpose(1, 2) * sh_batch[..., 22]).squeeze(2) +
                              C4[7] * ((xz * (xx - 3 * yy)).transpose(1, 2) * sh_batch[..., 23]).squeeze(2) +
                              C4[8] * ((xx * (xx - 3 * yy) - yy * (3 * xx - yy)).transpose(1, 2) * sh_batch[..., 24]).squeeze(2))

    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def qvec2rotmat(qvec):
    R = [[1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]]
    # check if is tensor
    if isinstance(qvec, torch.Tensor):
        return torch.tensor(R, dtype=qvec.dtype, device=qvec.device)
    else:
        return np.array(R)


def qvec2rotmat_batch(qvecs):
    """
    Convert a batch of quaternion vectors to rotation matrices.
    
    Args:
        qvecs: torch.Tensor of shape (batch_size, 4) containing quaternion vectors.
               Each quaternion is in the form [w, x, y, z].
    
    Returns:
        torch.Tensor of shape (batch_size, 3, 3) containing rotation matrices.
    """
    assert qvecs.shape[-1] == 4, "Each quaternion must have 4 components (w, x, y, z)."
    
    w, x, y, z = qvecs[:, 0], qvecs[:, 1], qvecs[:, 2], qvecs[:, 3]
    
    # Compute the rotation matrices
    R = torch.zeros((qvecs.shape[0], 3, 3), dtype=qvecs.dtype, device=qvecs.device)
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    
    return R


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def fit_ellipsoid_pca(points):
    """Fit ellipsoid surface parameters using an improved PCA method"""
    # Compute covariance matrix
    points = points.T
    mean = np.mean(points, axis=1, keepdims=True)
    centered_points = points - mean
    cov = np.cov(centered_points)

    # PCA decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Ensure a right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, -1] *= -1

    # Transform point cloud to the principal-axis frame
    points_in_pca = eigenvectors.T @ centered_points

    # Compute scale per axis (max distance to origin)
    scale = np.array([
        np.max(np.abs(points_in_pca[0])),
        np.max(np.abs(points_in_pca[1])),
        np.max(np.abs(points_in_pca[2]))
    ])

    # Convert rotation matrix to quaternion
    qvec = rotmat2qvec(eigenvectors)

    return scale, qvec, eigenvectors


def generate_ellipsoid_surface(center, R, abc, n_phi=30, n_theta=60):
    # Create a spherical coordinate grid
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta = np.linspace(0, np.pi, n_theta)
    phi, theta = np.meshgrid(phi, theta)

    # Compute Cartesian coordinates of ellipsoid surface points
    x = abc[0] * np.sin(theta) * np.cos(phi)
    y = abc[1] * np.sin(theta) * np.sin(phi)
    z = abc[2] * np.cos(theta)

    # Apply rotation and translation
    points = np.array([x.flatten(), y.flatten(), z.flatten()])
    rotated_points = R @ points
    x_rot = rotated_points[0, :].reshape(x.shape) + center[0]
    y_rot = rotated_points[1, :].reshape(y.shape) + center[1]
    z_rot = rotated_points[2, :].reshape(z.shape) + center[2]

    return x_rot, y_rot, z_rot


def ellipsoid_xyz2dirs(points, center, R, abc):
    X_centered = points - center
    # Rotate into the ellipsoid principal-axis frame
    X_rot = X_centered @ R
    # Divide by (a,b,c) to map to the unit sphere
    X_sph = X_rot / abc  # shape(N,3)

    x = X_sph[:, 0]
    y = X_sph[:, 1]
    z = X_sph[:, 2]
    r = np.sqrt(x*x + y*y + z*z)  # Ideally ~ 1
    return X_sph


def build_sh_basis(dirs, deg):
    assert deg <= 4 and deg >= 0
    M = dirs.shape[0]
    N = (deg + 1)**2

    x = dirs[:, 0]
    y = dirs[:, 1]
    z = dirs[:, 2]

    # Quadratic/cubic terms for later use
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    yz = y * z
    xz = x * z

    B = np.zeros((M, N), dtype=np.float64)

    # --- index 0
    # C0 * sh[..., 0]
    B[:, 0] = C0

    # If deg > 0, add indices 1, 2, 3
    idx = 1
    if deg >= 1:
        # index 1: -C1 * y
        B[:, idx] = -C1 * y; idx += 1
        # index 2:  C1 * z
        B[:, idx] =  C1 * z; idx += 1
        # index 3: -C1 * x
        B[:, idx] = -C1 * x; idx += 1

    # If deg > 1, add indices 4..8
    if deg >= 2:
        # index 4:  C2[0] * x*y
        B[:, idx] = C2[0] * xy; idx += 1
        # index 5:  C2[1] * y*z
        B[:, idx] = C2[1] * yz; idx += 1
        # index 6:  C2[2] * (2zz - xx - yy)
        B[:, idx] = C2[2] * (2.0 * zz - xx - yy); idx += 1
        # index 7:  C2[3] * x*z
        B[:, idx] = C2[3] * xz; idx += 1
        # index 8:  C2[4] * (xx - yy)
        B[:, idx] = C2[4] * (xx - yy); idx += 1

    # If deg > 2, add indices 9..15
    if deg >= 3:
        # index 9:   C3[0] * y*(3xx - yy)
        B[:, idx] = C3[0] * y * (3.0*xx - yy); idx += 1
        # index 10:  C3[1] * x*y*z
        B[:, idx] = C3[1] * x * y * z; idx += 1
        # index 11:  C3[2] * y*(4zz - xx - yy)
        B[:, idx] = C3[2] * y * (4.0*zz - xx - yy); idx += 1
        # index 12:  C3[3] * z*(2zz - 3xx - 3yy)
        B[:, idx] = C3[3] * z * (2.0*zz - 3.0*xx - 3.0*yy); idx += 1
        # index 13:  C3[4] * x*(4zz - xx - yy)
        B[:, idx] = C3[4] * x * (4.0*zz - xx - yy); idx += 1
        # index 14:  C3[5] * z*(xx - yy)
        B[:, idx] = C3[5] * z * (xx - yy); idx += 1
        # index 15:  C3[6] * x*(xx - 3yy)
        B[:, idx] = C3[6] * x * (xx - 3.0*yy); idx += 1

    # If deg > 3, add indices 16..24
    if deg >= 4:
        # index 16: C4[0] * x*y*(xx - yy)
        B[:, idx] = C4[0] * x * y * (xx - yy); idx += 1
        # index 17: C4[1] * y*z*(3xx - yy)
        B[:, idx] = C4[1] * y * z * (3.0*xx - yy); idx += 1
        # index 18: C4[2] * x*y*(7zz - 1)
        B[:, idx] = C4[2] * x * y * (7.0*zz - 1.0); idx += 1
        # index 19: C4[3] * y*z*(7zz - 3)
        B[:, idx] = C4[3] * y * z * (7.0*zz - 3.0); idx += 1
        # index 20: C4[4] * (zz*(35zz - 30) + 3)
        B[:, idx] = C4[4] * (zz*(35.0*zz - 30.0) + 3.0); idx += 1
        # index 21: C4[5] * x*z*(7zz - 3)
        B[:, idx] = C4[5] * x * z * (7.0*zz - 3.0); idx += 1
        # index 22: C4[6] * (xx - yy)*(7zz - 1)
        B[:, idx] = C4[6] * (xx - yy) * (7.0*zz - 1.0); idx += 1
        # index 23: C4[7] * x*z*(xx - 3yy)
        B[:, idx] = C4[7] * x * z * (xx - 3.0*yy); idx += 1
        # index 24: C4[8] * [ xx*(xx - 3yy) - yy*(3xx - yy) ]
        #          = C4[8] * (x^4 - 3x^2y^2 - 3x^2y^2 + y^4) = ...
        B[:, idx] = C4[8] * (xx*(xx - 3.0*yy) - yy*(3.0*xx - yy)); idx += 1

    return B


def fit_sh(dirs, vals, deg=3):
    B = build_sh_basis(dirs, deg)  # (M, N)

    # Check for NaNs or infinities in B and vals
    if np.any(np.isnan(B)) or np.any(np.isinf(B)):
        raise ValueError("B contains NaNs or infinities")
    if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
        raise ValueError("vals contains NaNs or infinities")

    # Check the shapes of B and vals
    if B.shape[0] != vals.shape[0]:
        raise ValueError(f"Shape mismatch: B has {B.shape[0]} rows, but vals has {vals.shape[0]} rows")

    # Use NumPy least squares
    # Note: if vals shape=(M,C), returns sh shape=(N,C).
    sh, residuals, rank, s = np.linalg.lstsq(B, vals, rcond=None)

    return sh


def sigmoid_inverse(y):
    return np.log(y / (1 - y))


def normalize_gaussian_param(data, mean=None, max=None, min=None):
    mean = [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -4.49235276e+00,
            -4.49503302e+00, -4.49633074e+00,  3.00459208e-01,  6.51899814e-05,
            1.89170773e-03 , 1.89177954e-03 , 2.49720841e+00 , 4.58014785e-03,
            4.46499611e-03 ,-1.14110523e-03 , 2.68523015e-04 ,-1.23834880e-04,
            2.35732927e-04 , 9.30548320e-05 ,-1.73384285e-04 ,-2.73907226e-04,
            2.23390361e-04 ,-8.36664910e-06 ,-8.09182054e-05 , 2.25439033e-04,
            2.26422015e-04 ,-1.97530483e-04 , 1.63569985e-05 ,-2.09461150e-04,
            1.69405925e-04 , 3.07092432e-04 ,-3.12061893e-04 ,-1.52989999e-05,
            -2.45888148e-06,  5.28432567e-05, -1.16569635e-04,  8.27079510e-06,
            -1.06544245e-04, -4.41882831e-04,  2.43246870e-04,  8.09790520e-05,
            -4.53681547e-05,  8.97818247e-05, -1.24178407e-04,  1.13853738e-05,
            2.47357192e-04 , 5.77839877e-05 , 1.44952501e-04 , 1.43054028e-05,
            1.01607328e-04 , 1.03135695e-04 ,-8.50049794e-05 ,-1.39808653e-05,
            -1.78836326e-05,  4.03624908e-05, -1.11568838e-04,  1.08969990e-04,
            -6.95070182e-05, -2.66065242e-05, -4.61056133e-05,] if mean is None else mean
    max = [ 0.        ,  0.        ,  0.        , -1.00015401, -1.00004894, -1.00000818,
            0.70709854,  0.70698323,  0.70634392,  0.99998619,  9.99965733,  4.21243114,
            4.48779123,  4.47898733,  0.2058102 ,  0.21131759,  0.24417596,  0.2034083,
            0.21486922,  0.20261809,  0.21203802,  0.24087037,  0.21596411,  0.21421831,
            0.21205019,  0.2110244 ,  0.23886215,  0.23158624,  0.21602105,  0.20716949,
            0.2178901 ,  0.20283779,  0.23275794,  0.2184898 ,  0.20906746,  0.2051165,
            0.22253056,  0.21163523,  0.23104285,  0.20371791,  0.25931755,  0.20722298,
            0.21982774,  0.23167768,  0.23026243,  0.22022465,  0.21967405,  0.22244092,
            0.21746668,  0.21545751,  0.21000853,  0.20300666,  0.21196338,  0.20343871,
            0.22452512,  0.22747277,  0.24362416,  0.21507951,  0.22239658,] if max is None else max
    min = [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 ,-7.99985032e+00 ,
            -7.99997354e+00, -7.99995510e+00,  5.41825135e-06, -7.06963142e-01,
            -7.05292962e-01, -9.99469842e-01, -4.99997138e+00, -4.42825116e+00,
            -5.35230080e+00, -4.08414391e+00, -2.10287899e-01, -2.09471978e-01,
            -2.12858210e-01, -2.10581176e-01, -2.11222921e-01, -2.14482616e-01,
            -2.16026399e-01, -1.99843370e-01, -2.32746661e-01, -2.14739236e-01,
            -1.89660265e-01, -2.23152858e-01, -2.07444540e-01, -2.18701150e-01,
            -2.49721778e-01, -2.14635715e-01, -2.23376562e-01, -2.47220994e-01,
            -2.18181762e-01, -2.10936668e-01, -2.07082077e-01, -2.21388228e-01,
            -2.03989228e-01, -2.17021142e-01, -2.13086043e-01, -2.39121230e-01,
            -2.23195444e-01, -2.19318503e-01, -2.09502482e-01, -2.18431642e-01,
            -2.11413467e-01, -2.52119369e-01, -2.06782913e-01, -2.25885522e-01,
            -2.41282672e-01, -2.24715470e-01, -2.09150451e-01, -2.23467359e-01,
            -2.12490952e-01, -2.02187655e-01, -2.13429589e-01, -2.36649322e-01,
            -2.04193526e-01, -2.13701161e-01, -2.12307555e-01,] if min is None else min

    mean = np.array(mean)[np.newaxis, :]
    max = np.array(max)[np.newaxis, :]
    min = np.array(min)[np.newaxis, :]

    # normalize to -1, 1 (except centroid)
    normalized = 2 * (data[..., 3:] - mean[..., 3:]) / (max[..., 3:] - min[..., 3:] + 1e-8) - 1
    out = np.concatenate([data[..., :3], normalized], axis=1)

    return out


def unnormalize_gaussian_param(data, mean=None, max=None, min=None):
    mean = [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -4.49235276e+00,
            -4.49503302e+00, -4.49633074e+00,  3.00459208e-01,  6.51899814e-05,
            1.89170773e-03 , 1.89177954e-03 , 2.49720841e+00 , 4.58014785e-03,
            4.46499611e-03 ,-1.14110523e-03 , 2.68523015e-04 ,-1.23834880e-04,
            2.35732927e-04 , 9.30548320e-05 ,-1.73384285e-04 ,-2.73907226e-04,
            2.23390361e-04 ,-8.36664910e-06 ,-8.09182054e-05 , 2.25439033e-04,
            2.26422015e-04 ,-1.97530483e-04 , 1.63569985e-05 ,-2.09461150e-04,
            1.69405925e-04 , 3.07092432e-04 ,-3.12061893e-04 ,-1.52989999e-05,
            -2.45888148e-06,  5.28432567e-05, -1.16569635e-04,  8.27079510e-06,
            -1.06544245e-04, -4.41882831e-04,  2.43246870e-04,  8.09790520e-05,
            -4.53681547e-05,  8.97818247e-05, -1.24178407e-04,  1.13853738e-05,
            2.47357192e-04 , 5.77839877e-05 , 1.44952501e-04 , 1.43054028e-05,
            1.01607328e-04 , 1.03135695e-04 ,-8.50049794e-05 ,-1.39808653e-05,
            -1.78836326e-05,  4.03624908e-05, -1.11568838e-04,  1.08969990e-04,
            -6.95070182e-05, -2.66065242e-05, -4.61056133e-05,] if mean is None else mean
    max = [ 0.        ,  0.        ,  0.        , -1.00015401, -1.00004894, -1.00000818,
            0.70709854,  0.70698323,  0.70634392,  0.99998619,  9.99965733,  4.21243114,
            4.48779123,  4.47898733,  0.2058102 ,  0.21131759,  0.24417596,  0.2034083,
            0.21486922,  0.20261809,  0.21203802,  0.24087037,  0.21596411,  0.21421831,
            0.21205019,  0.2110244 ,  0.23886215,  0.23158624,  0.21602105,  0.20716949,
            0.2178901 ,  0.20283779,  0.23275794,  0.2184898 ,  0.20906746,  0.2051165,
            0.22253056,  0.21163523,  0.23104285,  0.20371791,  0.25931755,  0.20722298,
            0.21982774,  0.23167768,  0.23026243,  0.22022465,  0.21967405,  0.22244092,
            0.21746668,  0.21545751,  0.21000853,  0.20300666,  0.21196338,  0.20343871,
            0.22452512,  0.22747277,  0.24362416,  0.21507951,  0.22239658,] if max is None else max
    min = [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 ,-7.99985032e+00 ,
            -7.99997354e+00, -7.99995510e+00,  5.41825135e-06, -7.06963142e-01,
            -7.05292962e-01, -9.99469842e-01, -4.99997138e+00, -4.42825116e+00,
            -5.35230080e+00, -4.08414391e+00, -2.10287899e-01, -2.09471978e-01,
            -2.12858210e-01, -2.10581176e-01, -2.11222921e-01, -2.14482616e-01,
            -2.16026399e-01, -1.99843370e-01, -2.32746661e-01, -2.14739236e-01,
            -1.89660265e-01, -2.23152858e-01, -2.07444540e-01, -2.18701150e-01,
            -2.49721778e-01, -2.14635715e-01, -2.23376562e-01, -2.47220994e-01,
            -2.18181762e-01, -2.10936668e-01, -2.07082077e-01, -2.21388228e-01,
            -2.03989228e-01, -2.17021142e-01, -2.13086043e-01, -2.39121230e-01,
            -2.23195444e-01, -2.19318503e-01, -2.09502482e-01, -2.18431642e-01,
            -2.11413467e-01, -2.52119369e-01, -2.06782913e-01, -2.25885522e-01,
            -2.41282672e-01, -2.24715470e-01, -2.09150451e-01, -2.23467359e-01,
            -2.12490952e-01, -2.02187655e-01, -2.13429589e-01, -2.36649322e-01,
            -2.04193526e-01, -2.13701161e-01, -2.12307555e-01,] if min is None else min

    mean = np.array(mean)[np.newaxis, :]
    max = np.array(max)[np.newaxis, :]
    min = np.array(min)[np.newaxis, :]

    unnormalized = (data[..., 3:] + 1) / 2 * (max[..., 3:] - min[..., 3:]) + mean[..., 3:]
    out = np.concatenate([data[..., :3], unnormalized], axis=1)

    return out


def rotmat2qvec_batch(R_batch):
    """
    Convert a batch of rotation matrices to quaternion vectors.
    
    Args:
        R_batch: torch.Tensor of shape (B, 3, 3) containing rotation matrices.
    
    Returns:
        torch.Tensor of shape (B, 4) containing quaternion vectors in [w, x, y, z] format.
    """
    B = R_batch.shape[0]
    device = R_batch.device
    dtype = R_batch.dtype
    
    # Extract rotation matrix elements for each batch
    Rxx = R_batch[:, 0, 0]  # (B,)
    Ryx = R_batch[:, 1, 0]  # (B,)
    Rzx = R_batch[:, 2, 0]  # (B,)
    Rxy = R_batch[:, 0, 1]  # (B,)
    Ryy = R_batch[:, 1, 1]  # (B,)
    Rzy = R_batch[:, 2, 1]  # (B,)
    Rxz = R_batch[:, 0, 2]  # (B,)
    Ryz = R_batch[:, 1, 2]  # (B,)
    Rzz = R_batch[:, 2, 2]  # (B,)
    
    # Build K matrix for each batch: (B, 4, 4)
    K_batch = torch.zeros((B, 4, 4), device=device, dtype=dtype)
    
    # Fill K matrices
    K_batch[:, 0, 0] = Rxx - Ryy - Rzz
    K_batch[:, 1, 0] = Ryx + Rxy
    K_batch[:, 1, 1] = Ryy - Rxx - Rzz
    K_batch[:, 2, 0] = Rzx + Rxz
    K_batch[:, 2, 1] = Rzy + Ryz
    K_batch[:, 2, 2] = Rzz - Rxx - Ryy
    K_batch[:, 3, 0] = Ryz - Rzy
    K_batch[:, 3, 1] = Rzx - Rxz
    K_batch[:, 3, 2] = Rxy - Ryx
    K_batch[:, 3, 3] = Rxx + Ryy + Rzz
    
    # Divide by 3.0
    K_batch = K_batch / 3.0
    
    # Batched eigendecomposition
    eigvals_batch, eigvecs_batch = torch.linalg.eigh(K_batch)  # (B, 4), (B, 4, 4)
    
    # Find the largest eigenvalue index for each batch
    max_indices = torch.argmax(eigvals_batch, dim=1)  # (B,)
    
    # Extract corresponding eigenvectors and reorder to [w, x, y, z]
    batch_indices = torch.arange(B, device=device)
    qvec_batch = torch.zeros((B, 4), device=device, dtype=dtype)
    
    # Reorder from [3, 0, 1, 2] indexing (w, x, y, z)
    qvec_batch[:, 0] = eigvecs_batch[batch_indices, 3, max_indices]  # w
    qvec_batch[:, 1] = eigvecs_batch[batch_indices, 0, max_indices]  # x
    qvec_batch[:, 2] = eigvecs_batch[batch_indices, 1, max_indices]  # y
    qvec_batch[:, 3] = eigvecs_batch[batch_indices, 2, max_indices]  # z
    
    # Ensure positive w component for each quaternion
    negative_w_mask = qvec_batch[:, 0] < 0
    qvec_batch[negative_w_mask] *= -1
    
    return qvec_batch


def stable_rotmat2qvec_batch(R_batch):
    """
    Numerically stable batched rotation-matrix to quaternion (w,x,y,z).
    Assumes R is proper (det ~ 1).
    """
    # R: (B,3,3)
    B = R_batch.shape[0]
    device = R_batch.device
    dtype = R_batch.dtype

    q = torch.empty((B, 4), device=device, dtype=dtype)

    trace = R_batch[:, 0, 0] + R_batch[:, 1, 1] + R_batch[:, 2, 2]

    # Masks
    mask0 = trace > 0
    mask1 = (~mask0) & (R_batch[:, 0, 0] >= R_batch[:, 1, 1]) & (R_batch[:, 0, 0] >= R_batch[:, 2, 2])
    mask2 = (~mask0) & (~mask1) & (R_batch[:, 1, 1] >= R_batch[:, 2, 2])
    mask3 = (~mask0) & (~mask1) & (~mask2)

    # Case 0
    t0 = torch.sqrt(trace[mask0] + 1.0) * 2.0  # 4*w
    q[mask0, 0] = 0.25 * t0
    q[mask0, 1] = (R_batch[mask0, 2, 1] - R_batch[mask0, 1, 2]) / t0
    q[mask0, 2] = (R_batch[mask0, 0, 2] - R_batch[mask0, 2, 0]) / t0
    q[mask0, 3] = (R_batch[mask0, 1, 0] - R_batch[mask0, 0, 1]) / t0

    # Case 1
    t1 = torch.sqrt(1.0 + R_batch[mask1, 0, 0] - R_batch[mask1, 1, 1] - R_batch[mask1, 2, 2]) * 2.0
    q[mask1, 0] = (R_batch[mask1, 2, 1] - R_batch[mask1, 1, 2]) / t1
    q[mask1, 1] = 0.25 * t1
    q[mask1, 2] = (R_batch[mask1, 0, 1] + R_batch[mask1, 1, 0]) / t1
    q[mask1, 3] = (R_batch[mask1, 0, 2] + R_batch[mask1, 2, 0]) / t1

    # Case 2
    t2 = torch.sqrt(1.0 - R_batch[mask2, 0, 0] + R_batch[mask2, 1, 1] - R_batch[mask2, 2, 2]) * 2.0
    q[mask2, 0] = (R_batch[mask2, 0, 2] - R_batch[mask2, 2, 0]) / t2
    q[mask2, 1] = (R_batch[mask2, 0, 1] + R_batch[mask2, 1, 0]) / t2
    q[mask2, 2] = 0.25 * t2
    q[mask2, 3] = (R_batch[mask2, 1, 2] + R_batch[mask2, 2, 1]) / t2

    # Case 3
    t3 = torch.sqrt(1.0 - R_batch[mask3, 0, 0] - R_batch[mask3, 1, 1] + R_batch[mask3, 2, 2]) * 2.0
    q[mask3, 0] = (R_batch[mask3, 1, 0] - R_batch[mask3, 0, 1]) / t3
    q[mask3, 1] = (R_batch[mask3, 0, 2] + R_batch[mask3, 2, 0]) / t3
    q[mask3, 2] = (R_batch[mask3, 1, 2] + R_batch[mask3, 2, 1]) / t3
    q[mask3, 3] = 0.25 * t3

    # Enforce w >= 0 for uniqueness
    neg = q[:, 0] < 0
    q[neg] = -q[neg]

    # Normalize
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return q

def canonicalize_rotation_batch(R_batch):
    """
    Make PCA rotation matrices deterministic by fixing column signs.
    For each column, find the row with largest absolute value; force that entry positive.
    Then fix det>0.
    """
    B = R_batch.shape[0]
    R = R_batch
    for col in range(3):
        col_vec = R[:, :, col]
        idx = torch.argmax(col_vec.abs(), dim=1)  # (B,)
        signs = torch.sign(col_vec[torch.arange(B), idx])
        signs[signs == 0] = 1.0
        R[:, :, col] = col_vec * signs.view(-1, 1)
    # Ensure right-handed
    det = torch.det(R)
    flip = det < 0
    if flip.any():
        R[flip, :, 2] *= -1
    return R


def softmax_max_batched(x, dim=-1, alpha=20.0):
    """
    Batched smooth approximation to max(x)
    x: (..., N) where ... can be batch dimensions
    """
    w = torch.softmax(alpha * x, dim=dim)
    return (w * x).sum(dim=dim)


def fit_ellipsoid_pca_torch_batched(points_batch, eps=1e-6, softmax_alpha=20.0,
                                    sort_descending=False, canonicalize=True, use_float64=True):
    """
    Enhanced: optional sorting & canonicalization and internal float64 for stability.
    """
    orig_dtype = points_batch.dtype
    if use_float64 and points_batch.dtype != torch.float64:
        points_batch = points_batch.double()
    B, N, _ = points_batch.shape
    mean_batch = torch.mean(points_batch, dim=1, keepdim=True)
    Xc_batch = points_batch - mean_batch
    Xc_T = Xc_batch.transpose(-2, -1)
    C_batch = torch.bmm(Xc_T, Xc_batch) / max(N - 1, 1)
    eye = torch.eye(3, device=points_batch.device, dtype=points_batch.dtype)
    C_batch = C_batch + eps * eye.unsqueeze(0)

    evals_batch, evecs_batch = torch.linalg.eigh(C_batch)  # ascending
    # Optionally sort descending (largest variance first)
    if sort_descending:
        idx = torch.argsort(evals_batch, dim=1, descending=True)
        evals_batch = torch.gather(evals_batch, 1, idx)
        # Reorder eigenvectors
        idx = idx.unsqueeze(1).expand(-1, 3, -1)  # Expand idx for batched gather
        evecs_batch = torch.gather(evecs_batch, 2, idx)

    R_batch = evecs_batch  # columns = axes
    if canonicalize:
        R_batch = canonicalize_rotation_batch(R_batch)

    # Project
    Xp_batch = torch.bmm(Xc_batch, R_batch)
    absXp = Xp_batch.abs()
    a = absXp[:, :, 0].max(dim=1).values # no gradient through max
    b = absXp[:, :, 1].max(dim=1).values # no gradient through max
    c = absXp[:, :, 2].max(dim=1).values # no gradient through max
    # a = softmax_max_batched(absXp[:, :, 0], dim=1, alpha=softmax_alpha) # with gradient
    # b = softmax_max_batched(absXp[:, :, 1], dim=1, alpha=softmax_alpha) # with gradient
    # c = softmax_max_batched(absXp[:, :, 2], dim=1, alpha=softmax_alpha) # with gradient
    abc_batch = torch.stack([a, b, c], dim=1) + eps

    # Cast back if needed
    if orig_dtype != R_batch.dtype:
        R_batch = R_batch.to(orig_dtype)
        abc_batch = abc_batch.to(orig_dtype)
        mean_batch = mean_batch.to(orig_dtype)

    return abc_batch, R_batch, mean_batch.squeeze(1)


def ellipsoid_xyz2dirs_torch_batched(points_batch, center_batch, R_batch, abc_batch):
    """
    Batched conversion from points to ellipsoid directions
    points_batch: (B, N, 3)
    center_batch: (B, 3)
    R_batch: (B, 3, 3)
    abc_batch: (B, 3)
    Returns: dirs_batch (B, N, 3)
    """
    # Center the points: (B, N, 3)
    Xc_batch = points_batch - center_batch.unsqueeze(1)
    
    # Rotate to ellipsoid frame: (B, N, 3)
    Xr_batch = torch.bmm(Xc_batch, R_batch)
    
    # Normalize by ellipsoid radii: (B, N, 3)
    dirs_batch = Xr_batch / abc_batch.unsqueeze(1)
    
    return dirs_batch


def build_sh_basis_torch_batched(dirs_batch, deg=3):
    """
    Batched SH basis construction
    dirs_batch: (B, N, 3) assumed unit-ish directions
    Returns: B_batch (B, N, (deg+1)^2)
    """
    B, N, _ = dirs_batch.shape
    x, y, z = dirs_batch[:, :, 0], dirs_batch[:, :, 1], dirs_batch[:, :, 2]  # (B, N)
    
    # Use global constants
    basis_list = []
    
    # Degree 0: C0 constant
    basis_list.append(torch.full((B, N, 1), C0, device=dirs_batch.device, dtype=dirs_batch.dtype))
    
    if deg >= 1:
        # Degree 1 terms
        basis_list.extend([
            (-C1 * y).unsqueeze(-1),  # (B, N, 1)
            (C1 * z).unsqueeze(-1),
            (-C1 * x).unsqueeze(-1)
        ])
    
    if deg >= 2:
        # Degree 2 terms
        xx, yy, zz = x*x, y*y, z*z
        xy, yz, xz = x*y, y*z, x*z
        basis_list.extend([
            (C2[0] * xy).unsqueeze(-1),
            (C2[1] * yz).unsqueeze(-1),
            (C2[2] * (2*zz - xx - yy)).unsqueeze(-1),
            (C2[3] * xz).unsqueeze(-1),
            (C2[4] * (xx - yy)).unsqueeze(-1)
        ])
    
    if deg >= 3:
        # Degree 3 terms
        basis_list.extend([
            (C3[0] * y * (3*xx - yy)).unsqueeze(-1),
            (C3[1] * x * y * z).unsqueeze(-1),
            (C3[2] * y * (4*zz - xx - yy)).unsqueeze(-1),
            (C3[3] * z * (2*zz - 3*xx - 3*yy)).unsqueeze(-1),
            (C3[4] * x * (4*zz - xx - yy)).unsqueeze(-1),
            (C3[5] * z * (xx - yy)).unsqueeze(-1),
            (C3[6] * x * (xx - 3*yy)).unsqueeze(-1)
        ])
    
    return torch.cat(basis_list, dim=-1)  # (B, N, (deg+1)^2)


def fit_sh_torch_batched(dirs_batch, vals_batch, deg=3, eps=1e-4):
    """
    Batched SH fitting
    dirs_batch: (B, N, 3)
    vals_batch: (B, N, C) e.g., RGB colors
    Returns: sh_batch ((B, (deg+1)^2, C))
    """
    B, N, C = vals_batch.shape
    
    # Build SH basis: (B, N, K) where K = (deg+1)^2
    B_batch = build_sh_basis_torch_batched(dirs_batch, deg=deg)
    K = B_batch.shape[-1]
    
    # Batched least squares: solve (B^T B + eps I) x = B^T vals for each batch
    BT_batch = B_batch.transpose(-2, -1)  # (B, K, N)
    
    # Compute normal equations: (B, K, K)
    normal_batch = torch.bmm(BT_batch, B_batch)
    
    # Add regularization
    eye = torch.eye(K, device=dirs_batch.device, dtype=dirs_batch.dtype)
    normal_batch = normal_batch + eps * eye.unsqueeze(0).expand(B, -1, -1)
    
    # Right hand side: (B, K, C)
    rhs_batch = torch.bmm(BT_batch, vals_batch)
    
    # Solve using Cholesky decomposition
    try:
        L_batch = torch.linalg.cholesky(normal_batch)  # (B, K, K)
        sh_batch = torch.cholesky_solve(rhs_batch, L_batch)  # (B, K, C)
    except RuntimeError:
        # Fallback to LU decomposition if Cholesky fails
        sh_batch = torch.linalg.solve(normal_batch, rhs_batch)
    
    return sh_batch


def point2gaussian_torch_batched(data_batch, deg=3, softmax_alpha=100.0, eps=1e-6):
    """
    Updated to use stable PCA + stable quaternion + channel-contiguous SH ordering parity.
    """
    if data_batch.dim() == 2:
        data_batch = data_batch.unsqueeze(0)
    B, N, _ = data_batch.shape
    points_batch = data_batch[:, :, :3]
    colors_batch = data_batch[:, :, 3:6]
    opac_batch = data_batch[:, :, 6]

    # Stable PCA fit
    abc_batch, R_batch, center_batch = fit_ellipsoid_pca_torch_batched(
        points_batch, eps=eps, softmax_alpha=softmax_alpha,
        sort_descending=True, canonicalize=True, use_float64=False
    )

    # Stable quaternion
    q_batch = stable_rotmat2qvec_batch(R_batch)

    # Directions
    dirs_batch = ellipsoid_xyz2dirs_torch_batched(points_batch, center_batch, R_batch, abc_batch)
    # Optionally renormalize tiny drift
    dirs_norm = dirs_batch.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    dirs_batch = dirs_batch / dirs_norm

    # SH fit
    sh_batch = fit_sh_torch_batched(dirs_batch, colors_batch, deg=deg)  # (B,K,3)

    # Coefficient ordering: DC separate, then per-channel (R all, G all, B all) excluding DC
    feat_dc_batch = sh_batch[:, 0, :]                      # (B,3)
    if sh_batch.shape[1] > 1:
        rest = sh_batch[:, 1:, :]                          # (B,K-1,3)
        feat_rest_r = rest[:, :, 0]
        feat_rest_g = rest[:, :, 1]
        feat_rest_b = rest[:, :, 2]
        feat_extra_batch = torch.cat([feat_rest_r, feat_rest_g, feat_rest_b], dim=1)
    else:
        feat_extra_batch = torch.zeros((B, 0), device=sh_batch.device, dtype=sh_batch.dtype)

    scale_batch = torch.log(abc_batch.clamp_min(1e-6))
    m_batch = opac_batch.mean(dim=1).clamp(1e-6, 1 - 1e-6)
    opacity_batch = torch.log(m_batch / (1 - m_batch))

    return torch.cat([
        center_batch,
        scale_batch,
        q_batch,
        opacity_batch.unsqueeze(1),
        feat_dc_batch,
        feat_extra_batch
    ], dim=1)


def gaussian2point_torch_batch(gaussians, num_points=144, batch_size=1024, 
                               clip_color=False, device='cuda'):
    total_gaussians = len(gaussians)
    results = []

    for i in range(0, total_gaussians, batch_size):
        batch_gaussians = gaussians[i:i + batch_size]
        current_batch_size = len(batch_gaussians)
        
        # Move data to GPU and batch it
        centroids = torch.stack([torch.tensor(g.centroid, device=device) for g in batch_gaussians])
        scales = torch.stack([torch.tensor(g.scale, device=device) for g in batch_gaussians])
        rotations = torch.stack([torch.tensor(g.rotation, device=device) for g in batch_gaussians])
        opacities = torch.stack([torch.tensor(g.opacity, device=device) for g in batch_gaussians])
        feat_dcs = torch.stack([torch.tensor(g.feat_dc, device=device) for g in batch_gaussians])
        feat_extras = torch.stack([torch.tensor(g.feat_extra, device=device) for g in batch_gaussians])
        
        # Activate parameters (vectorized)
        scales = torch.exp(scales)
        rotations = rotations / torch.norm(rotations, dim=1, keepdim=True)
        # opacities = torch.sigmoid(opacities)
        opacities = 1 / (1 + torch.exp(-opacities))
        
        # Convert rotations to rotation matrices (batched)
        R_batch = qvec2rotmat_batch(rotations)  # [batch, 3, 3]
        
        # Prepare spherical harmonics coefficients (batched)
        extra_len = feat_extras.shape[1] // 3
        coeff_r = torch.cat([feat_dcs[:, 0:1], feat_extras[:, :extra_len]], dim=1)
        coeff_g = torch.cat([feat_dcs[:, 1:2], feat_extras[:, extra_len:2*extra_len]], dim=1)
        coeff_b = torch.cat([feat_dcs[:, 2:3], feat_extras[:, 2*extra_len:]], dim=1)
        sh_batch = torch.stack([coeff_r, coeff_g, coeff_b], dim=2)  # [batch, coeffs, 3]
        sh_batch = sh_batch.permute(0, 2, 1)  # [batch, 3, coeffs]
        
        # Generate UV coordinates (same for all Gaussians)
        size = int(np.sqrt(num_points))
        u_ = torch.linspace(0, 2 * np.pi, size, device=device)
        v_ = torch.linspace(0, np.pi, size, device=device)
        u, v = torch.meshgrid(u_, v_, indexing='ij')
        
        # Process all Gaussians in the current batch
        x_batch, y_batch, z_batch, color_batch = _uv_to_xyz_color_torch_batch(
            u, v, scales, R_batch, sh_batch, device=device
        )
        
        if clip_color:
            color_batch = torch.clip(color_batch + 0.5, min=0.0, max=1.0)
        
        results.append((x_batch, y_batch, z_batch, color_batch, opacities))
    
    # Combine results from all batches
    x, y, z, color, opacity = zip(*results)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    z = torch.cat(z, dim=0)
    color = torch.cat(color, dim=0)
    opacity = torch.cat(opacity, dim=0)

    x = x.view(-1, num_points).unsqueeze(-1)
    y = y.view(-1, num_points).unsqueeze(-1) 
    z = z.view(-1, num_points).unsqueeze(-1)
    color = color.permute(0, 2, 1)
    opacity = opacity.repeat(1, num_points).unsqueeze(-1)
    points = torch.cat([x, y, z, color, opacity], dim=-1)
    
    return points


def _uv_to_xyz_color_torch_batch(u, v, scales, R_batch, sh_batch, device='cuda'):
    """Batched version of _uv_to_xyz_color_torch"""
    batch_size = scales.shape[0]
    
    # Generate sphere coordinates
    x = torch.sin(v) * torch.cos(u)
    y = torch.sin(v) * torch.sin(u)
    z = torch.cos(v)
    
    # Expand for batch processing
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    num_points = len(x_flat)
    
    # Scale and rotate points (batched)
    points = torch.stack([
        scales[:, 0:1] * x_flat.unsqueeze(0).expand(batch_size, -1),
        scales[:, 1:2] * y_flat.unsqueeze(0).expand(batch_size, -1),
        scales[:, 2:3] * z_flat.unsqueeze(0).expand(batch_size, -1)
    ], dim=2)  # [batch, points, 3]
    
    # Apply rotation: [batch, 3, 3] @ [batch, points, 3] -> [batch, points, 3]
    rotated_points = torch.bmm(points, R_batch.transpose(1, 2))
    
    x_rot = rotated_points[:, :, 0].reshape(batch_size, *x.shape)
    y_rot = rotated_points[:, :, 1].reshape(batch_size, *y.shape)
    z_rot = rotated_points[:, :, 2].reshape(batch_size, *z.shape)
    
    # Prepare directions for SH evaluation
    dirs = torch.stack([x_flat, y_flat, z_flat], dim=1)  # [points, 3]
    dirs = dirs.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, points, 3]
    
    # Evaluate spherical harmonics (batched)
    colors = eval_sh_batch(3, sh_batch, dirs)
    
    return x_rot, y_rot, z_rot, colors
