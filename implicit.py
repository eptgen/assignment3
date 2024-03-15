import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd

from ray_utils import RayBundle


# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(
            points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = torch.abs(points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)

# Torus SDF class
class TorusSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.radii.val).float().unsqueeze(0), requires_grad=cfg.radii.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack(
            [
                torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
                diff[..., -1],
            ],
            dim=-1
        )
        return (torch.linalg.norm(q, dim=-1) - self.radii[..., 1]).unsqueeze(-1)

sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
    'torus': TorusSDF,
}


# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle.sample_points)
        density = self._sdf_to_density(signed_distance)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }

        return out


# Converts SDF into density/feature volume
class SDFSurface(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )
        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )
    
    def get_distance(self, points):
        points = points.view(-1, 3)
        return self.sdf(points)

    def get_color(self, points):
        points = points.view(-1, 3)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        return base_color * self.feature * points.new_ones(points.shape[0], 1)
    
    def forward(self, points):
        return self.get_distance(points)

class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y

"""
# legacy neuralradiancefield (better?)
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim
        # print("shapes pts/dirs", embedding_dim_xyz, embedding_dim_dir)
        
        self.xyz_hidden = nn.Linear(embedding_dim_xyz, cfg.n_hidden_neurons_xyz)
        self.to_density = nn.Linear(cfg.n_hidden_neurons_xyz, 1)
        self.relu = nn.ReLU()
        self.to_feature_vector = nn.Linear(cfg.n_hidden_neurons_xyz, 256)
        total_feature = 256 + embedding_dim_dir
        
        self.dir_hidden = nn.Linear(total_feature, cfg.n_hidden_neurons_dir)
        
        self.to_color = nn.Linear(cfg.n_hidden_neurons_dir, 3)
        self.sig = nn.Sigmoid()
        
        # view independent
        self.to_color_ind = nn.Linear(cfg.n_hidden_neurons_xyz, 3)
        self.to_density_ind = nn.Linear(cfg.n_hidden_neurons_xyz, 1)

    def forward(self, ray_bundle):
        
        # view dependent
        pts = ray_bundle.sample_points # (H*W, n_points, 3)
        n_points = pts.shape[1]
        pts = pts.reshape(-1, 3) # (B, 3)
        dirs = (ray_bundle.directions.unsqueeze(1) * torch.ones(n_points, 1, device = "cuda")).reshape(-1, 3) # (B, 3)
        pts = self.harmonic_embedding_xyz(pts) # (B, hexyz_output_dim)
        dirs = self.harmonic_embedding_dir(dirs) # (B, hedir_output_dim)
        pts = self.xyz_hidden(pts) # (B, hidden_xyz)
        sigma = self.to_density(pts) # (B, 1)
        sigma = self.relu(sigma) # (B, 1)
        
        features = self.to_feature_vector(pts) # (B, 256)
        both = torch.cat((features, dirs), dim = 1) # (B, 256 + hedir_output_dim)
        both = self.dir_hidden(both) # (B, hidden_dir)
        color = self.to_color(both) # (B, 3)
        color = self.sig(color) # (B, 3)
        
        # view independent
        pts = ray_bundle.sample_points # (H*W, n_points, 3)
        pts = pts.reshape(-1, 3) # (B, 3)
        pts = self.harmonic_embedding_xyz(pts) # (B, hexyz_output_dim)
        pts = self.xyz_hidden(pts) # (B, hidden_xyz)
        color = self.to_color_ind(pts) # (B, 3)
        color = self.sig(color)
        sigma = self.to_density_ind(pts) # (B, 1)
        sigma = self.relu(sigma)
        
        return {"feature": color, "density": sigma}
        
"""
"""
# legacy 8-layer
# TODO (Q3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        self.hidden_1 = nn.Linear(embedding_dim_xyz, 256)
        self.relu1 = nn.ReLU()
        self.hidden_2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.hidden_3 = nn.Linear(256, 256)
        self.relu3 = nn.ReLU()
        self.hidden_4 = nn.Linear(256, 256)
        self.relu4 = nn.ReLU()
        self.hidden_5 = nn.Linear(256, 256)
        self.relu5 = nn.ReLU()
        self.hidden_6 = nn.Linear(256 + embedding_dim_xyz, 256)
        self.relu6 = nn.ReLU()
        self.hidden_7 = nn.Linear(256, 256)
        self.relu7 = nn.ReLU()
        self.hidden_8 = nn.Linear(256, 256)
        self.relu8 = nn.ReLU()
        self.hidden_9 = nn.Linear(256, 256)
        self.relu9 = nn.ReLU()
        self.to_density = nn.Linear(256, 1)
        self.relu10 = nn.ReLU()
        
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim
        self.hidden_10 = nn.Linear(256 + embedding_dim_dir, 128)
        self.relu11 = nn.ReLU()
        # print("shapes pts/dirs", embedding_dim_xyz, embedding_dim_dir)
        
        self.to_color = nn.Linear(128, 3)
        self.sig = nn.Sigmoid()
        
        # view independent
        self.hidden_10_ind = nn.Linear(256, 128)

    def forward(self, ray_bundle):
        # view dependent
        
        pts = ray_bundle.sample_points # (H*W, n_points, 3)
        n_points = pts.shape[1]
        pts = pts.reshape(-1, 3) # (B, 3)
        pts = self.harmonic_embedding_xyz(pts) # (B, hexyz_output_dim)
        features = self.hidden_1(pts) # (B, 256)
        features = self.relu1(features) # (B, 256)
        features = self.hidden_2(features) # (B, 256)
        features = self.relu2(features) # (B, 256)
        features = self.hidden_3(features) # (B, 256)
        features = self.relu3(features) # (B, 256)
        features = self.hidden_4(features) # (B, 256)
        features = self.relu4(features) # (B, 256)
        features = self.hidden_5(features) # (B, 256)
        features = self.relu5(features) # (B, 256)
        features = torch.cat((features, pts), dim = 1) # (B, 256 + hexyz_output_dim)
        features = self.hidden_6(features) # (B, 256)
        features = self.relu6(features) # (B, 256)
        features = self.hidden_7(features) # (B, 256)
        features = self.relu7(features) # (B, 256)
        features = self.hidden_8(features) # (B, 256)
        features = self.relu8(features) # (B, 256)
        features_nosigma = self.hidden_9(features) # (B, 256)
        features_nosigma = self.relu9(features_nosigma) # (B, 256)
        sigma = self.to_density(features) # (B, 1)
        sigma = self.relu10(sigma) # (B, 1)
        dirs = (ray_bundle.directions.unsqueeze(1) * torch.ones(n_points, 1, device = "cuda")).reshape(-1, 3) # (B, 3)
        dirs = self.harmonic_embedding_dir(dirs) # (B, hedir_output_dim)
        features_nosigma = torch.cat((features_nosigma, dirs), dim = 1) # (B, 256 + hedir_output_dim)
        features_nosigma = self.hidden_10(features_nosigma) # (B, 128)
        features_nosigma = self.relu11(features_nosigma) # (B, 128)
        color = self.to_color(features_nosigma) # (B, 3)
        color = self.sig(color) # (B, 3)
        
        # view independent
        pts = ray_bundle.sample_points # (H*W, n_points, 3)
        pts = pts.reshape(-1, 3) # (B, 3)
        pts = self.harmonic_embedding_xyz(pts) # (B, hexyz_output_dim)
        features = self.hidden_1(pts) # (B, 256)
        features = self.relu1(features) # (B, 256)
        features = self.hidden_2(features) # (B, 256)
        features = self.relu2(features) # (B, 256)
        features = self.hidden_3(features) # (B, 256)
        features = self.relu3(features) # (B, 256)
        features = self.hidden_4(features) # (B, 256)
        features = self.relu4(features) # (B, 256)
        features = self.hidden_5(features) # (B, 256)
        features = self.relu5(features) # (B, 256)
        features = torch.cat((features, pts), dim = 1) # (B, 256 + hexyz_output_dim)
        features = self.hidden_6(features) # (B, 256)
        features = self.relu6(features) # (B, 256)
        features = self.hidden_7(features) # (B, 256)
        features = self.relu7(features) # (B, 256)
        features = self.hidden_8(features) # (B, 256)
        features = self.relu8(features) # (B, 256)
        features_nosigma = self.hidden_9(features) # (B, 256)
        features_nosigma = self.relu9(features_nosigma) # (B, 256)
        sigma = self.to_density(features) # (B, 1)
        sigma = self.relu10(sigma) # (B, 1)
        features_nosigma = self.hidden_10_ind(features_nosigma) # (B, 128)
        features_nosigma = self.relu11(features_nosigma) # (B, 128)
        color = self.to_color(features_nosigma) # (B, 3)
        color = self.sig(color) # (B, 3)
        
        return {"feature": color, "density": sigma}
"""

"""
# View independent
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        self.n_layers_xyz = cfg.n_layers_xyz
        hidden_neurons_xyz = cfg.n_hidden_neurons_xyz
        self.append_xyz = cfg.append_xyz
        
        layers = [[]]
        num_seq = 0
        for i in range(self.n_layers_xyz):
            fc_in = hidden_neurons_xyz
            if i == 0:
                fc_in = embedding_dim_xyz
            if i in self.append_xyz:
                fc_in = hidden_neurons_xyz + embedding_dim_xyz
                layers.append([])
                num_seq += 1
            fc_out = hidden_neurons_xyz
            if i == self.n_layers_xyz - 1:
                fc_out = hidden_neurons_xyz + 1
            layers[num_seq].append(nn.Linear(fc_in, fc_out, device = "cuda"))
            if i != self.n_layers_xyz - 1: layers[num_seq].append(nn.ReLU())
         
        self.layers = torch.nn.Sequential(*[torch.nn.Sequential(*layer) for layer in layers])
        
        self.relu_sigma = nn.ReLU()
        self.to_color = nn.Linear(hidden_neurons_xyz, 3, device = "cuda")
        self.sigmoid_color = nn.Sigmoid()

    def forward(self, ray_bundle):
        pts = ray_bundle.sample_points # (H*W, n_points, 3)
        n_points = pts.shape[1]
        pts = pts.view(-1, 3) # (B, 3)
        pts = self.harmonic_embedding_xyz(pts) # (B, hexyz_output_dim)
        features = pts
        i = 0
        for layer in self.layers:
            features = layer(features)
            if i != len(self.layers) - 1: features = torch.cat((features, pts), dim = 1)
            i += 1
        sigma = features[:, 0] # (B, 1)
        sigma = self.relu_sigma(sigma).view(-1, n_points, 1) # (B, 1)
        color = self.to_color(features[:, 1:]) # (B, 3)
        color = self.sigmoid_color(color).view(-1, n_points, 3) # (B, 3)
        
        return {"feature": color, "density": sigma}
"""

# View dependent
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        self.n_layers_xyz = cfg.n_layers_xyz
        hidden_neurons_xyz = cfg.n_hidden_neurons_xyz
        self.append_xyz = cfg.append_xyz
        
        layers = [[]]
        num_seq = 0
        for i in range(self.n_layers_xyz):
            fc_in = hidden_neurons_xyz
            if i == 0:
                fc_in = embedding_dim_xyz
            if i in self.append_xyz:
                fc_in = hidden_neurons_xyz + embedding_dim_xyz
                layers.append([])
                num_seq += 1
            fc_out = hidden_neurons_xyz
            if i == self.n_layers_xyz - 1:
                fc_out = hidden_neurons_xyz + 1
            layers[num_seq].append(nn.Linear(fc_in, fc_out, device = "cuda"))
            if i != self.n_layers_xyz - 1: layers[num_seq].append(nn.ReLU())
         
        self.layers = torch.nn.Sequential(*[torch.nn.Sequential(*layer) for layer in layers])
        
        self.relu_sigma = nn.ReLU()
        
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim
        hidden_neurons_dir = cfg.n_hidden_neurons_dir
        self.fc_color = nn.Linear(hidden_neurons_xyz + embedding_dim_dir, hidden_neurons_dir, device = "cuda")
        self.relu_color = nn.ReLU()
        self.to_color = nn.Linear(hidden_neurons_dir, 3, device = "cuda")
        self.sigmoid_color = nn.Sigmoid()

    def forward(self, ray_bundle):
        pts = ray_bundle.sample_points # (H*W, n_points, 3)
        n_points = pts.shape[1]
        pts = pts.view(-1, 3) # (B, 3)
        pts = self.harmonic_embedding_xyz(pts) # (B, hexyz_output_dim)
        features = pts
        i = 0
        for layer in self.layers:
            features = layer(features)
            if i != len(self.layers) - 1: features = torch.cat((features, pts), dim = 1)
            i += 1
        sigma = features[:, 0] # (B, 1)
        sigma = self.relu_sigma(sigma).view(-1, n_points, 1) # (B, 1)
        dirs = (ray_bundle.directions.unsqueeze(1) * torch.ones(n_points, 1, device = "cuda")).reshape(-1, 3) # (B, 3)
        dirs = self.harmonic_embedding_dir(dirs) # (B, hedir_output_dim)
        color = torch.cat((features[:, 1:], dirs), dim = 1) # (B, hidden_dir + hedir_output_dim)
        color = self.fc_color(color) # (B, hidden_dir)
        color = self.relu_color(color) # (B, hidden_dir)
        color = self.to_color(color) # (B, 3)
        color = self.sigmoid_color(color).view(-1, n_points, 3) # (B, 3)
        
        return {"feature": color, "density": sigma}

class NeuralSurface(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        self.n_layers_dist = cfg.n_layers_distance
        self.append_dist = cfg.append_distance
        hidden_neurons_dist = cfg.n_hidden_neurons_distance
        
        layers = [[]]
        num_seq = 0
        for i in range(self.n_layers_dist):
            fc_in = hidden_neurons_dist
            if i == 0:
                fc_in = embedding_dim_xyz
            if i in self.append_dist:
                fc_in = hidden_neurons_dist + embedding_dim_xyz
                layers.append([])
                num_seq += 1
            fc_out = hidden_neurons_dist
            if i == self.n_layers_dist - 1:
                fc_out = hidden_neurons_dist + 1
            layers[num_seq].append(nn.Linear(fc_in, fc_out, device = "cuda"))
            layers[num_seq].append(nn.ReLU())
        
        self.layers = torch.nn.Sequential(*[torch.nn.Sequential(*layer) for layer in layers])
        self.to_sd = nn.Linear(hidden_neurons_dist, hidden_neurons_dist + 1, device = "cuda")
        
        self.n_layers_color = cfg.n_layers_color
        hidden_neurons_color = cfg.n_hidden_neurons_color
        
        layers_color = []
        for i in range(self.n_layers_color):
            fc_in = hidden_neurons_color
            if i == 0:
                fc_in = hidden_neurons_dist
            fc_out = hidden_neurons_color
            layers_color.append(nn.Linear(fc_in, fc_out, device = "cuda"))
            layers_color.append(nn.ReLU())
            
        self.layers_color = nn.Sequential(*layers_color)
        self.to_color = nn.Linear(hidden_neurons_color, 3, device = "cuda")
        self.sigmoid_color = nn.Sigmoid()

    def get_distance(
        self,
        points
    ):
        '''
        TODO: Q6
        Output:
            distance: N X 1 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3) # (B, 3)
        points = self.harmonic_embedding_xyz(points) # (B, hedist_dim)
        features = points
        i = 0
        for layer in self.layers:
            features = layer(features)
            if i != len(self.layers) - 1: features = torch.cat((features, points), dim = 1)
            i += 1
        sds = features[:, 0] # (B, 1)
        return sds
    
    def get_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance: N X 3 Tensor, where N is number of input points
        '''
        return self.get_distance_color(points)[1]
    
    def get_distance_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance, points: N X 1, N X 3 Tensors, where N is number of input points
        You may just implement this by independent calls to get_distance, get_color
            but, depending on your MLP implementation, it maybe more efficient to share some computation
        '''
        points = points.view(-1, 3) # (B, 3)
        points = self.harmonic_embedding_xyz(points) # (B, hedist_dim)
        features = points
        i = 0
        for layer in self.layers:
            features = layer(features)
            if i != len(self.layers) - 1: features = torch.cat((features, points), dim = 1)
            i += 1
        features = self.to_sd(features) # (B, hidden_dist + 1)
        sds = features[:, 0] # (B, 1)
        colors = features[:, 1:] # (B, hidden_dist)
        colors = self.layers_color(colors) # (B, hidden_color)
        colors = self.to_color(colors) # (B, 3)
        colors = self.sigmoid_color(colors) # (B, 3)
        return (sds, colors)
        
    def forward(self, points):
        return self.get_distance(points)

    def get_distance_and_gradient(
        self,
        points
    ):
        has_grad = torch.is_grad_enabled()
        points = points.view(-1, 3)

        # Calculate gradient with respect to points
        with torch.enable_grad():
            points = points.requires_grad_(True)
            distance = self.get_distance(points)
            gradient = autograd.grad(
                distance,
                points,
                torch.ones_like(distance, device=points.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        
        return distance, gradient


implicit_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
    'sdf_surface': SDFSurface,
    'neural_surface': NeuralSurface,
}
