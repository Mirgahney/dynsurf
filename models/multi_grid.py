import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Sequential as Seq, Linear, ReLU, Parameter
from models.smooth_sampler import SmoothSampler
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import knn_graph
from networkx import grid_graph
from torch_geometric.utils import convert
from scipy.stats import norm
import numpy as np
import math
from pdb import set_trace as st


def generated_grid_data(res=100):
    num_x, num_y, num_z = (res, res, res)
    G = grid_graph(dim=(num_x, num_y, num_z))
    data = convert.from_networkx(G)
    #data = G

    #x = np.linspace(-0.5, 0.5, num_x)
    #y = np.linspace(-0.5, 0.5, num_y)
    #z = np.linspace(-0.5, 0.5, num_z)
    #xv, yv, zv = np.meshgrid(x, y, z)

    #data.x = torch.from_numpy(np.column_stack((xv.flatten(), yv.flatten(), zv.flatten()))).float()

    return data.edge_index


def generated_grid_alldata(start_res=35, pg_scale=[2, 2]):
    edge_indices = []
    G = grid_graph(dim=(start_res, start_res, start_res))
    data = convert.from_networkx(G)
    edge_indices.append(data.edge_index)
    res = start_res
    for scale in pg_scale:
        res = res * scale
        G = grid_graph(dim=(res, res, res))
        data = convert.from_networkx(G)
        edge_indices.append(data.edge_index)
    return edge_indices


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
#class GraphGaussianBlur:
    def __init__(self, std:float=1.0, aggr:str="add"):
        super().__init__(aggr=aggr)
        self.std = std
        self.g_weight = norm.pdf(0.00294, 0.0, self.std)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return 0.125*x_j + 0.125*x_i
        #return (gaussian(x_i, x_j)[:, None]*x_j) + 0.125*x_i


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)


class DEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6):
        super().__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, batch=None):
        #st()
        edge_index = knn_graph(x, self.k, loop=False, flow=self.flow)
        return super().forward(x, edge_index)


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


class GeometricMultiGrid(nn.Module):
    def __init__(self, sizes, initial_value, out_feature_dim):
        super().__init__()
        volumes = []
        self.sizes = sizes
        self.ggb = None

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
        self.edge_indices = [ei.to('cuda') for ei in generated_grid_alldata(self.vres, [2, 2])]
        self.scale_step = 0

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
        #self.edge_index = generated_grid_data(self.vres).to('cuda')
        self.scale_step += 1
        self.edge_index = self.edge_indices[self.scale_step].to('cuda')


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
        self.edge_indices = [ei.to('cuda') for ei in generated_grid_alldata(self.vres, [2, 2])]
        self.scale_step = 0

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
        #self.edge_index = generated_grid_data(self.vres).to('cuda')
        self.scale_step += 1
        self.edge_index = self.edge_indices[self.scale_step].to('cuda')


class GCNMultiGrid(nn.Module):
    def __init__(self, sizes, initial_value, out_feature_dim):
        super().__init__()
        volumes = []
        self.sizes = sizes
        #st()
        self.ggb = GCNConv(out_feature_dim[0], out_feature_dim[0])

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
        self.edge_indices = [ei.to('cuda') for ei in generated_grid_alldata(self.vres, [2, 2])]
        self.scale_step = 0

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
            volume_x = volume.permute((1, 2, 3, 0))
            volume_dims = volume_x.shape[-1]
            volume_x = volume_x.reshape(-1, volume_dims)
            volume = self.ggb(volume_x, self.edge_index)
            volume = volume.reshape((self.vres, self.vres, self.vres, volume_dims)).unsqueeze(0)
            volume = volume.permute((0, 4, 1, 2, 3))
            upscale_volume = F.interpolate(volume.data, size=tuple(world_size), mode='trilinear', align_corners=True)
            volumes.append(nn.Parameter(upscale_volume))
        self.volumes = nn.ParameterList(volumes)
        self.vres = world_sizes[0][0].item()
        #self.edge_index = generated_grid_data(self.vres).to('cuda')
        self.scale_step += 1
        self.edge_index = self.edge_indices[self.scale_step].to('cuda')


class EdgeCNMultiGrid(nn.Module):
    def __init__(self, sizes, initial_value, out_feature_dim):
        super().__init__()
        volumes = []
        self.sizes = sizes
        #st()
        self.ggb = EdgeConv(out_feature_dim[0], out_feature_dim[0])

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
        self.edge_indices = [ei.to('cuda') for ei in generated_grid_alldata(self.vres, [2, 2])]
        self.scale_step = 0

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
        #self.edge_index = generated_grid_data(self.vres).to('cuda')
        self.scale_step += 1
        self.edge_index = self.edge_indices[self.scale_step].to('cuda')


class DEdgeCNMultiGrid(GeometricMultiGrid):
    def __init__(self, sizes, initial_value, out_feature_dim):
        super().__init__(sizes, initial_value, out_feature_dim)
        #st()
        self.ggb = DEdgeConv(out_feature_dim[0], out_feature_dim[0])

    def forward(self, grid, **args):
        return super().forward(grid, **args)
