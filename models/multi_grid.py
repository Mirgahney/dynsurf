import torch
from torch import nn
from torch.nn import functional as F
from models.smooth_sampler import SmoothSampler
from torch_geometric.nn import MessagePassing
from networkx import grid_graph
from torch_geometric.utils import convert
from scipy.stats import norm
import numpy as np
import math
from pdb import set_trace


def generated_grid_data(res=100):
    num_x, num_y, num_z = (res, res, res)
    G = grid_graph(dim=(num_x, num_y, num_z))
    data = convert.from_networkx(G)

    #x = np.linspace(-0.5, 0.5, num_x)
    #y = np.linspace(-0.5, 0.5, num_y)
    #z = np.linspace(-0.5, 0.5, num_z)
    #xv, yv, zv = np.meshgrid(x, y, z)

    #data.x = torch.from_numpy(np.column_stack((xv.flatten(), yv.flatten(), zv.flatten()))).float()

    return data.edge_index


def gaussian(x=0, y=0, sigma=1):
    return 1 / (2*math.pi*sigma) * \
    torch.exp(-((x - y).norm(dim=-1) / (2*sigma**2)))


def sample_volumes(volumes, grid, align_corners=True, apply_smoothstep=False, concat=True):
    out = []

    for volume in volumes:
        out.append(SmoothSampler.apply(volume.contiguous(), grid, align_corners, apply_smoothstep))

    if concat:
        return torch.cat(out, dim=1)

    return out


class GraphGaussianBlur(MessagePassing):
    def __init__(self, std:float=1.0, aggr:str="add"):
        super().__init__(aggr=aggr)
        self.std = std
        self.g_weight = norm.pdf(0.00294, 0.0, self.std)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return 0.125*x_j + 0.125*x_i
        #return (gaussian(x_i, x_j)[:, None]*x_j) + 0.125*x_i


class MultiGrid(nn.Module):
    def __init__(self, sizes, initial_value, out_feature_dim):
        super().__init__()
        volumes = []
        self.sizes = sizes

        prev_bound = 0

        for i, size in enumerate(sizes):
            next_bound = prev_bound + out_feature_dim[i]
            init_val = initial_value[:, prev_bound:next_bound]
            prev_bound = next_bound
            volume = nn.Parameter(init_val.detach().view(1, -1, 1, 1, 1).repeat(1, 1, *size))
            volumes.append(volume)

        self.volumes = nn.ParameterList(volumes)

    def forward(self, grid, **args):
        return sample_volumes(self.volumes, grid, **args)

    # TODO: upscale 2 times
    def upscale(self, world_sizes):
        volumes = []
        for i, (volume, world_size) in enumerate(zip(self.volumes, world_sizes)):
            upscale_volume = F.interpolate(volume.data, size=tuple(world_size), mode='trilinear', align_corners=True)
            volumes.append(nn.Parameter(upscale_volume))
        self.volumes = nn.ParameterList(volumes)


class GaussianMultiGrid(nn.Module):
    def __init__(self, sizes, initial_value, out_feature_dim):
        super().__init__()
        volumes = []
        self.sizes = sizes
        self.ggb = GraphGaussianBlur(aggr="mean")

        prev_bound = 0

        for i, size in enumerate(sizes):
            next_bound = prev_bound + out_feature_dim[i]
            init_val = initial_value[:, prev_bound:next_bound]
            prev_bound = next_bound
            volume = nn.Parameter(init_val.detach().view(1, -1, 1, 1, 1).repeat(1, 1, *size))
            volumes.append(volume)

        self.volumes = nn.ParameterList(volumes)
        self.vres = self.sizes[0][0].item()
        self.edge_index = generated_grid_data(self.vres).to('cuda')

    def forward(self, grid, **args):
        volume_x = self.volumes[0].squeeze(0).permute((1,2,3,0))
        volume_dims = volume_x.shape[-1]
        volume_x = volume_x.reshape(-1, volume_dims)
        volumes = self.ggb(volume_x, self.edge_index)
        volumes = volumes.reshape((self.vres, self.vres, self.vres, volume_dims)).unsqueeze(0)
        volumes = volumes.permute((0, 4, 1, 2, 3))
        return sample_volumes([volumes], grid, **args)

    # TODO: upscale 2 times
    def upscale(self, world_sizes):
        volumes = []
        for i, (volume, world_size) in enumerate(zip(self.volumes, world_sizes)):
            upscale_volume = F.interpolate(volume.data, size=tuple(world_size), mode='trilinear', align_corners=True)
            volumes.append(nn.Parameter(upscale_volume))
        self.volumes = nn.ParameterList(volumes)
        self.vres = world_sizes[0][0].item()
        self.edge_index = generated_grid_data(self.vres).to('cuda')
