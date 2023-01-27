import torch
from torch import nn


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, feat_volume):
        #feat_volume = feat_volume.squeeze(0)
        batch_size = feat_volume.size()[0]
        x_v = feat_volume.size()[2]
        y_v = feat_volume.size()[3]
        z_v = feat_volume.size()[4]
        count_x = self._tensor_size(feat_volume[:, :, 1:, :, :])
        count_y = self._tensor_size(feat_volume[:, :, :, 1:, :])
        count_z = self._tensor_size(feat_volume[:, :, :, :, 1:])
        x_tv = torch.pow((feat_volume[:, :, 1:, :, :] - feat_volume[:, :, :x_v-1, :, :]), 2).sum()
        y_tv = torch.pow((feat_volume[:, :, :, 1:, :] - feat_volume[:, :, :, :y_v-1, :]), 2).sum()
        z_tv = torch.pow((feat_volume[:, :, :, :, 1:] - feat_volume[:, :, :, :, :z_v-1]), 2).sum()

        return self.TVLoss_weight * 2 * (x_tv/count_x + y_tv/count_y + z_tv/count_z)/batch_size

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]*t.size()[4]