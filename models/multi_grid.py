import torch
from torch import nn
from torch.nn import functional as F
from models.smooth_sampler import SmoothSampler
from pdb import set_trace


def sample_volumes(volumes, grid, align_corners=True, apply_smoothstep=False, concat=True):
    out = []

    for volume in volumes:
        out.append(SmoothSampler.apply(volume.contiguous(), grid, align_corners, apply_smoothstep))

    if concat:
        return torch.cat(out, dim=1)

    return out


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
