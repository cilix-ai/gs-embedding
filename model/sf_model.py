import torch
import torch.nn as nn
import torch.nn.functional as F


class ChamferLoss(nn.Module):

    def __init__(self, geo_weight=0.5):
        super(ChamferLoss, self).__init__()
        self.geo_weight = geo_weight


    def forward(self, x, y):
        return self.chamfer_distance_with_color(x, y, alpha=self.geo_weight)


    def chamfer_distance(self, x, y):
        """
        Computes the Chamfer Distance between two point sets x and y.
        Args:
            x: Tensor of shape [batch_size, num_points, 3]
            y: Tensor of shape [batch_size, num_points, 3]
        Returns:
            Chamfer Distance: Scalar tensor representing the batch-wise distance
        """
        x = x.unsqueeze(2)  # [batch_size, num_points_x, 1, 3]
        y = y.unsqueeze(1)  # [batch_size, 1, num_points_y, 3]

        # Squared Euclidean distance between each point in x and y
        dist = torch.sum((x - y) ** 2, dim=-1)  # [batch_size, num_points_x, num_points_y]

        # Find minimum distance from each point in x to y, and from y to x
        min_dist_x_to_y = torch.min(dist, dim=2)[0]  # [batch_size, num_points_x]
        min_dist_y_to_x = torch.min(dist, dim=1)[0]  # [batch_size, num_points_y]

        # Mean distance for both directions
        loss_x_to_y = torch.mean(min_dist_x_to_y)
        loss_y_to_x = torch.mean(min_dist_y_to_x)

        # Total Chamfer Distance
        chamfer_loss = loss_x_to_y + loss_y_to_x

        return chamfer_loss


    def chamfer_distance_with_color(self, x, y, alpha=0.5, tau=-0.01):
        """
        Computes the Chamfer Distance between two point sets x and y, including color information.
        Args:
            x: Tensor of shape [batch_size, num_points, 6], where each point has (x, y, z, r, g, b)
            y: Tensor of shape [batch_size, num_points, 6], where each point has (x, y, z, r, g, b)
            alpha: Weight factor to balance between geometric and color distances (default is 0.5)
        Returns:
            Chamfer Distance: Scalar tensor representing the batch-wise distance including color matching
        """
        # Split coordinates and color components
        coords_x, color_x = x[:, :, :3], x[:, :, 3:]  # Coordinates: (x, y, z), Colors: (r, g, b)
        coords_y, color_y = y[:, :, :3], y[:, :, 3:]

        # Normalize by the maximum norm (optional for scaling)
        norm = torch.norm(coords_x, dim=-1).max(dim=-1, keepdim=True)[0].unsqueeze(2)

        # Compute geometric (x, y, z) Chamfer Distance
        coords_x = coords_x.unsqueeze(2)  # [batch_size, num_points_x, 1, 3]
        coords_y = coords_y.unsqueeze(1)  # [batch_size, 1, num_points_y, 3]
        geo_dist = torch.sqrt(torch.sum((coords_x - coords_y) ** 2, dim=-1)) / norm  # [batch_size, num_points_x, num_points_y]

        # Find minimum geometric distance and corresponding indices
        min_geo_dist_x_to_y, indices_x_to_y = torch.min(geo_dist, dim=2)  # [batch_size, num_points_x]
        min_geo_dist_y_to_x, indices_y_to_x = torch.min(geo_dist, dim=1)  # [batch_size, num_points_y]

        # Step 1: From geo_dist, find the nearest coords_y for each coords_x (min distance + index)
        min_dist, min_idx =  min_geo_dist_y_to_x, indices_y_to_x  # [B, N], min_dist is the minimum distance, min_idx is the nearest-point index
        # Use indices to fetch the nearest coords_y colors for each coords_x
        color_y_nearest = torch.gather(color_y, 1, indices_x_to_y.unsqueeze(-1).expand(-1, -1, color_y.size(-1)))  # [B, N, color_dim]
        color_dist1 = torch.sqrt(torch.sum((color_x - color_y_nearest)**2,dim=2))  # Compute color distance, result [B, N, color_dim]

        color_x_nearest = torch.gather(color_x, 1, min_idx.unsqueeze(-1).expand(-1, -1, color_x.size(-1)))  # [B, N, color_dim]
        color_dist2 = torch.sqrt(torch.sum((color_y - color_x_nearest)**2, dim=2))  # Compute color distance, result [B, N, color_dim]

        # Step 4: Weight color distances and average as loss
        weighted_color_dist = color_dist1.mean()+color_dist2.mean() #* weights.unsqueeze(-1)  # Multiply weights with color distances
        loss_color = weighted_color_dist#.mean()  # Average as final loss

        # Weighted sum of geometric and color distances
        combined_dist_x_to_y = min_geo_dist_x_to_y #+ (1 - alpha) * color_dist_x_to_y  # [batch_size, num_points_x]
        combined_dist_y_to_x =  min_geo_dist_y_to_x #+ (1 - alpha) * color_dist_y_to_x  # [batch_size, num_points_y]

        # Mean distance for both directions
        loss_x_to_y = torch.mean(combined_dist_x_to_y)
        loss_y_to_x = torch.mean(combined_dist_y_to_x)
        # Total Chamfer Distance with color matching
        chamfer_loss_with_color = alpha * loss_x_to_y + alpha * loss_y_to_x + (1 - alpha)* loss_color

        return chamfer_loss_with_color, alpha * loss_x_to_y + alpha * loss_y_to_x, (1 - alpha)* loss_color


class SFEncoder(nn.Module):

    def __init__(self, input_dim=3, feature_dim=512, vae=True, deterministic=False):
        super(SFEncoder, self).__init__()
        
        self.vae = vae
        self.deterministic = deterministic

        # MLP to extract local features
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, feature_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(feature_dim)

        if self.vae:
            # Fully connected layers for the global feature
            self.fc_mu = nn.Linear(feature_dim, feature_dim)
            self.fc_logvar = nn.Linear(feature_dim, feature_dim)
        else:
            self.fc1 = nn.Linear(feature_dim, feature_dim)
            self.fc2 = nn.Linear(feature_dim, feature_dim)


    def forward(self, x):
        # x shape: [batch_size, num_points, input_dim]
        x = x.transpose(2, 1)  # [batch_size, input_dim, num_points]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))  # [batch_size, feature_dim, num_points]

        # Max pooling to get the global feature
        x = torch.max(x, 2)[0]  # [batch_size, feature_dim]

        if self.vae:
            # Fully connected layers for mu and logvar
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            logvar = torch.clamp(logvar, min=-6, max=3)  # Clamping to avoid numerical issues

            # Reparameterization trick
            if self.deterministic:
                z = mu
            else:
                std = torch.exp(0.5 * logvar) + 1e-8
                z = mu + std * torch.randn_like(std)

            return z, mu, logvar
        else:
            # Fully connected layers
            x = F.relu(self.fc1(x))
            x = self.fc2(x)  # Global feature
            return x


class SFDecoder(nn.Module):

    def __init__(self, grid_dim=45, out_dim=3, feature_dim=512):
        super(SFDecoder, self).__init__()

        # Create the 2D grid for folding
        self.grid_dim = grid_dim

        u, v = torch.meshgrid(
            torch.linspace(0, 2 * torch.pi, grid_dim),  # u: azimuth (0 to 2*pi)
            torch.linspace(0, torch.pi, grid_dim)       # v: elevation (0 to pi)
        )
        x = torch.sin(v) * torch.cos(u) * 0.1
        y = torch.sin(v) * torch.sin(u) * 0.1
        z = torch.cos(v) * 0.1
        grid = torch.stack([x, y, z], dim=-1).view(-1, 3)

        self.register_buffer('grid', grid)

        # MLPs for folding
        self.fc1 = nn.Linear(self.grid.shape[-1] + feature_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 3)  # Output a 3D point

        self.fc1_c = nn.Linear(3 + feature_dim, 512)
        self.fc2_c = nn.Linear(512, 512)
        self.fc3_c = nn.Linear(512, out_dim-3)  # Output a 3D point


    def forward(self, global_feature):
        # Create the folding grid, shape [batch_size, grid_dim^2, 2]
        batch_size = global_feature.size(0)
        grid = self.grid.unsqueeze(0).repeat(batch_size, 1, 1)

        # Concatenate the global feature with the grid
        global_feature = global_feature.unsqueeze(1).repeat(1, self.grid_dim ** 2, 1)
        x = torch.cat([grid, global_feature], dim=2)  # [batch_size, grid_dim^2, 2 + feature_dim]

        # Pass through the folding MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # [batch_size, grid_dim^2, 3]

        x_ = x.clone().detach()
        color = torch.cat([x_, global_feature], dim=2)
        color = F.relu(self.fc1_c(color))
        color = F.relu(self.fc2_c(color))
        color = self.fc3_c(color)

        return torch.cat([x, color], dim=2)


class SFModel(nn.Module):

    def __init__(self, input_dim=3, feature_dim=32, grid_dim=24, vae=True, deterministic=False):
        super(SFModel, self).__init__()
        self.vae = vae
        self.encoder = SFEncoder(input_dim=input_dim, feature_dim=feature_dim, vae=vae, deterministic=deterministic)
        self.decoder = SFDecoder(grid_dim=grid_dim, out_dim=input_dim, feature_dim=feature_dim)


    def forward(self, x):
        if self.vae:
            z, mu, logvar = self.encoder(x)
            reconstructed_points = self.decoder(z)
            return reconstructed_points, z, mu, logvar
        else:
            global_feature = self.encoder(x)
            reconstructed_points = self.decoder(global_feature)
            return reconstructed_points, global_feature
