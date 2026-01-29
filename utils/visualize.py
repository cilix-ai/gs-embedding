import torch
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from gsplat import rasterization
import random
import math


def set_axes_equal(ax):
    """Set 3D plot axes to have equal scale."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = np.mean(limits, axis=1)
    max_range = np.max(limits[:, 1] - limits[:, 0])
    for ctr, lim in zip(centers, limits):
        ax.set_xlim3d([ctr - max_range / 2, ctr + max_range / 2])
        ax.set_ylim3d([ctr - max_range / 2, ctr + max_range / 2])
        ax.set_zlim3d([ctr - max_range / 2, ctr + max_range / 2])


def visualize_point_cloud(
        input: torch.Tensor,
        filename: str = 'point_cloud.png',
        axis_equal: bool = True,
        axis_off: bool = True):

    point_cloud, color = input[..., :3][0], input[..., 3:6][0]

    mx = color.max()
    mi = color.min()
    if mx > 1 or mi < 0:
        # normalize to [0, 1]
        color = (color - mi) / (mx - mi)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates
    x = point_cloud[:, 0].detach().cpu().numpy()
    y = point_cloud[:, 1].detach().cpu().numpy()
    z = point_cloud[:, 2].detach().cpu().numpy()

    # Handle colors if provided
    if color is not None:
        if color.shape[1] != 3:
            raise ValueError("Colors must have shape (N, 3) with values in the range [0, 1]")
        c = color.detach().cpu().numpy()
    else:
        c = 'b'  # Default color if no color is provided

    # Plot the point cloud
    ax.scatter(x, y, z, c=c, s=3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if axis_equal:
        set_axes_equal(ax)

    # turn off the axis
    if axis_off:
        ax.axis('off')

    # if path does not exist, create it
    # import os
    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)


def visualize_point_cloud_o3d(points, color=None, filename='point_cloud.png', window=False):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(color)

    if window:
        o3d.visualization.draw_geometries([point_cloud], window_name='point_cloud')
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(point_cloud)
        vis.update_geometry(point_cloud)
        view_control = vis.get_view_control()
        front = np.array([1., 1., 1.])
        front /= np.linalg.norm(front)
        view_control.set_front(front.tolist())
        view_control.set_lookat([0.0, 0.0, 0.0])
        view_control.set_up([0.0, 0.0, 1.0])
        view_control.set_zoom(0.8)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(filename)
        vis.destroy_window()


def get_view_matrix(eye, target, up):
    """
    Computes the world-to-camera (view) matrix.

    Parameters:
        eye:    (3,) array-like, camera position
        target: (3,) array-like, look-at point
        up:     (3,) array-like, up direction

    Returns:
        view_matrix: (4,4) numpy array
    """
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    # Forward vector (camera direction, points from target to eye)
    z = eye - target
    z /= np.linalg.norm(z)

    # Right vector
    x = np.cross(up, z)
    x /= np.linalg.norm(x)

    # True up vector
    y = np.cross(z, x)

    # Build rotation matrix
    rot = np.array([
        [x[0], y[0], z[0], 0],
        [x[1], y[1], z[1], 0],
        [x[2], y[2], z[2], 0],
        [0,    0,    0,    1]
    ])

    # Build translation matrix
    trans = np.array([
        [1, 0, 0, -eye[0]],
        [0, 1, 0, -eye[1]],
        [0, 0, 1, -eye[2]],
        [0, 0, 0, 1]
    ])

    # View matrix is rot * trans
    view_matrix = rot @ trans
    return view_matrix


def generate_random_view_matrix(device='cuda', 
                                distance_range=(2.0, 8.0), 
                                elevation_range=(-30, 60),
                                azimuth_range=(0, 360),
                                look_at=[0, 0, 0],
                                up_vector=[0, 0, 1]):
    """
    Generate a random view matrix
    """
    # Random spherical coordinates
    elevation = math.radians(random.uniform(elevation_range[0], elevation_range[1]))
    azimuth = math.radians(random.uniform(azimuth_range[0], azimuth_range[1]))
    distance = random.uniform(distance_range[0], distance_range[1])
    
    # Convert to camera position
    x = distance * math.cos(elevation) * math.cos(azimuth)
    y = distance * math.cos(elevation) * math.sin(azimuth)
    z = distance * math.sin(elevation)
    
    camera_position = [x, y, z]
    print(camera_position)
    
    # Generate view matrix using your existing function
    viewmat = get_view_matrix(camera_position, look_at, up_vector)
    return torch.tensor(viewmat, dtype=torch.float32, device=device).unsqueeze(0)


def visualize_gaussian(gaussians, 
                       camera_distance=2.0,
                       viewmat=None,
                       h=1080, w=1920, 
                       fov=60.0,
                       sh_degree=0,
                       white_bg=True,
                       save_img=False,
                       filename="gaussian.png"):
    """
    Input Gaussians format:
    N x [xyz, scales, rotations, opacities, sh_dc, sh_rest] = N x 59
    """
    assert isinstance(gaussians, torch.Tensor), "Input should be a tensor"
    assert gaussians.dim() == 2
    if torch.isnan(gaussians).any():
        gaussians = torch.nan_to_num(gaussians, nan=0.0)
    if torch.isinf(gaussians).any():
        gaussians = torch.where(torch.isinf(gaussians), torch.zeros_like(gaussians), gaussians)

    device = gaussians.device

    # camera intrinsics
    f = (w / 2) / torch.tan(torch.tensor(fov / 2, device=device) * torch.pi / 180)
    K = torch.tensor([[f, 0.0, w/2],
                      [0.0, f, h/2],
                      [0.0, 0.0, 1.0]],
                      dtype=torch.float32,
                      device=device
                      ).unsqueeze(0)

    centroids = gaussians[:, :3]
    # center the scene
    # centroids -= centroids.mean(dim=0)
    # offset scene to camera (used for mipnerf360)
    centroids += torch.tensor([0, 0, camera_distance], dtype=torch.float32, device=device)
    scales = torch.exp(gaussians[:, 3:6])
    rotations = gaussians[:, 6:10]
    opacities = torch.sigmoid(gaussians[:, 10])
    sh = gaussians[:, 11:14].unsqueeze(1)

    # world to camera
    if viewmat is None:
        # used for mipnerf360
        viewmat = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0) 
        # used for shapesplat
        # viewmat = get_view_matrix([0.9, 0, 0], [0, 0, 0], [0, 1, 0])
        # # rotate -90 degree along viewing direction
        # rotation_neg_90 = np.array([[0, 1, 0, 0],
        #                             [-1, 0, 0, 0],
        #                             [0, 0, 1, 0],
        #                             [0, 0, 0, 1]], dtype=np.float32)
        # viewmat = rotation_neg_90 @ viewmat
        # viewmat = torch.tensor(viewmat, dtype=torch.float32, device=device).unsqueeze(0)

    render_color, render_alpha, _ = rasterization(means=centroids, quats=rotations, scales=scales,
                                                  opacities=opacities, colors=sh, Ks=K, viewmats=viewmat,
                                                  width=w, height=h, sh_degree=sh_degree)

    render_color = render_color.squeeze(0)
    render_color = torch.clamp(render_color, min=0.0, max=1.0)
    if white_bg:
        render_alpha = render_alpha.squeeze(0)
        white_bg = torch.ones_like(render_color, device=device)
        render_color = render_color * render_alpha + white_bg * (1.0 - render_alpha)

    render_color = render_color.cpu().numpy()

    if save_img:
        plt.imsave(filename, render_color)

    return render_color
