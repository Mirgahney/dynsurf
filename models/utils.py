import torch
from torch import nn
#from torchtyping import TensorType
from math import floor, log
from rich.console import Console
from abc import abstractmethod
from typing import Optional


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1, steps=[1]):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.steps = steps

    def forward(self, feat_volume):
        #feat_volume = feat_volume.squeeze(0)
        batch_size = feat_volume.size()[0]
        x_v = feat_volume.size()[2]
        y_v = feat_volume.size()[3]
        z_v = feat_volume.size()[4]
        loss = 0.0
        for step in self.steps:
            idxs_a = torch.arange(1, x_v, step)
            idxs_b = torch.arange(0, x_v-1, step)
            loss = loss + self._get_diffs(feat_volume, idxs_a, idxs_b)

        return self.TVLoss_weight * 2 * loss/batch_size

    def _get_diffs(self, feat_volume, idxs_a, idxs_b):
        count_x = self._tensor_size(feat_volume[:, :, idxs_a, :, :])
        count_y = self._tensor_size(feat_volume[:, :, :, idxs_a, :])
        count_z = self._tensor_size(feat_volume[:, :, :, :, idxs_a])
        x_tv = torch.pow((feat_volume[:, :, idxs_a, :, :] - feat_volume[:, :, idxs_b, :, :]), 2).sum()
        y_tv = torch.pow((feat_volume[:, :, :, idxs_a, :] - feat_volume[:, :, :, idxs_b, :]), 2).sum()
        z_tv = torch.pow((feat_volume[:, :, :, :, idxs_a] - feat_volume[:, :, :, :, idxs_b]), 2).sum()
        return (x_tv/count_x + y_tv/count_y + z_tv/count_z)

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]*t.size()[4]


CONSOLE = Console(width=120)


def print_tcnn_speed_warning(method_name: str):
    """Prints a warning about the speed of the TCNN."""
    CONSOLE.line()
    CONSOLE.print(f"[bold yellow]WARNING: Using a slow implementation of {method_name}. ")
    CONSOLE.print(
        "[bold yellow]:person_running: :person_running: "
        + "Install tcnn for speedups :person_running: :person_running:"
    )
    CONSOLE.print("[yellow]pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch")
    CONSOLE.line()


def human_format(num):
    """Format a number in a more human readable way

    Args:
        num: number to format
    """
    units = ["", "K", "M", "B", "T", "P"]
    k = 1000.0
    magnitude = int(floor(log(num, k)))
    return f"{(num / k**magnitude):.2f} {units[magnitude]}"

"""
The field module baseclass.
"""


class FieldComponent(nn.Module):
    """Field modules that can be combined to store and compute the fields.

    Args:
        in_dim: Input dimension to module.
        out_dim: Ouput dimension to module.
    """

    def __init__(self, in_dim: Optional[int] = None, out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def build_nn_modules(self) -> None:
        """Function instantiates any torch.nn members within the module.
        If none exist, do nothing."""

    def set_in_dim(self, in_dim: int) -> None:
        """Sets input dimension of encoding

        Args:
            in_dim: input dimension
        """
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding."""
        if self.out_dim is None:
            raise ValueError("Output dimension has not been set")
        return self.out_dim

    @abstractmethod
    #def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
    def forward(self, in_tensor):

        """
    Returns processed tensor

    Args:
        in_tensor: Input tensor to process
    """
        raise NotImplementedError
