import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # print("rays/deltas shape", rays_density.shape, deltas.shape)
        """
        num_weights = deltas.shape[1]
        batch_size = deltas.shape[0]
        weights = torch.zeros(batch_size, num_weights, device = "cuda")
        T = torch.ones(batch_size, device = "cuda")
        rays_density_sq = rays_density.squeeze()
        deltas_sq = deltas.squeeze()
        for i in range(num_weights):
            prod = -rays_density_sq[:, i] * deltas_sq[:, i]
            # print("prod sum", torch.sum(prod))
            weights[:, i] = T * (1 - torch.exp(prod))
            T *= torch.exp(-rays_density_sq[:, i] * deltas_sq[:, i])
        """
        B = deltas.shape[0]
        n_points = deltas.shape[1]
        rays_density_sq = rays_density.squeeze() # (B, n_points)
        deltas_sq = deltas.squeeze() # (B, n_points)
        prods = -rays_density_sq * deltas_sq # (B, n_points)
        exp_prods = torch.exp(prods) # (B, n_points)
        one_minus_exp_prods = 1 - exp_prods # (B, n_points)
        T = torch.cumprod(torch.cat((torch.ones(B, device = "cuda").unsqueeze(1), exp_prods), dim = 1)[:, :-1], dim = 1) # (B, n_points)

        return T * one_minus_exp_prods # (B, n_points)
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # print("weights/rays shape", weights.shape, rays_feature.shape)
        return torch.sum(weights.unsqueeze(-1) * rays_feature.reshape(weights.shape[0], weights.shape[1], -1), dim = 1)

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []
        
        # print("B, chunk_size", B, self._chunk_size)

        for chunk_start in range(0, B, self._chunk_size):
            chunk_end = min(B, chunk_start+self._chunk_size)
            cur_ray_bundle = ray_bundle[chunk_start:chunk_end]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            # print(type(implicit_fn))
            # print(implicit_output.keys())
            density = implicit_output['density']
            feature = implicit_output['feature']
            # print("density", density.shape)
            # print("feature", feature.shape)
            # print("ray bundle", cur_ray_bundle.shape)
            # print("density L0", torch.norm(density, p = 0))

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            )
            # print("weights sum", torch.sum(weights))

            # TODO (1.5): Render (color) features using weights
            feature = self._aggregate(weights, feature)

            # TODO (1.5): Render depth map
            # print("sample_lengths shape", cur_ray_bundle.sample_lengths.shape)
            depth = self._aggregate(weights, cur_ray_bundle.sample_lengths)

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class SphereTracingRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self.near = cfg.near
        self.far = cfg.far
        self.max_iters = cfg.max_iters
    
    def sphere_tracing(
        self,
        implicit_fn,
        origins, # Nx3
        directions, # Nx3
    ):
        '''
        Input:
            implicit_fn: a module that computes a SDF at a query point
            origins: N_rays X 3
            directions: N_rays X 3
        Output:
            points: N_rays X 3 points indicating ray-surface intersections. For rays that do not intersect the surface,
                    the point can be arbitrary.
            mask: N_rays X 1 (boolean tensor) denoting which of the input rays intersect the surface.
        '''
        # TODO (Q5): Implement sphere tracing
        # 1) Iteratively update points and distance to the closest surface
        #   in order to compute intersection points of rays with the implicit surface
        # 2) Maintain a mask with the same batch dimension as the ray origins,
        #   indicating which points hit the surface, and which do not
        it = 0
        N = origins.shape[0]
        points = origins # (N, 3)
        mask = torch.ones(N, 1, dtype = torch.bool, device = "cuda") # (N, 1)
        while it < self.max_iters:
            dists = implicit_fn(points) # (N, 1)
            points += directions * dists
            mask = torch.logical_and(mask, dists < self.far)
            it += 1
        return (points, mask)

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]
            points, mask = self.sphere_tracing(
                implicit_fn,
                cur_ray_bundle.origins,
                cur_ray_bundle.directions
            )
            mask = mask.repeat(1,3)
            isect_points = points[mask].view(-1, 3)

            # Get color from implicit function with intersection points
            isect_color = implicit_fn.get_color(isect_points)

            # Return
            color = torch.zeros_like(cur_ray_bundle.origins)
            color[mask] = isect_color.view(-1)

            cur_out = {
                'color': color.view(-1, 3),
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta):
    def psi(s):
        leq_0 = 0.5 * torch.exp(s / beta)
        g_0 = 1 - 0.5 * torch.exp(-s / beta)
        return (s <= 0) * leq_0 + (s > 0) * g_0
    return alpha * psi(-signed_distance)

class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            density = sdf_to_density(distance, self.alpha, self.beta) # TODO (Q7): convert SDF to density

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'color': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer,
    'sphere_tracing': SphereTracingRenderer,
    'volume_sdf': VolumeSDFRenderer
}
