import torch
from torch import nn


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