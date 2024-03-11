import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray, device = "cuda")

        # TODO (Q1.4): Sample points from z values
        origins = ray_bundle.origins.unsqueeze(1) # shape: (H*W, 1, 3)
        unsqueezed_z = z_vals.unsqueeze(1) # shape: (n_points, 1)
        unsqueezed_ray_bundle = torch.nn.functional.normalize(ray_bundle.directions, dim = 1).unsqueeze(1) # shape: (H*W, 1, 3)
        sampled = unsqueezed_ray_bundle * unsqueezed_z # shape: (H*W, n_points, 3)
        # print("shapes", unsqueezed_z.shape, unsqueezed_ray_bundle.shape, sampled.shape, origins.shape)
        sample_points = sampled + origins # shape: (H*W, n_points, 3)

        # Return
        # print("sample_points shape", sample_points.shape)
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals.unsqueeze(1) * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}