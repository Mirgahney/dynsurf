import torch
import torch.nn as nn
import numpy as np
from models.embedder import get_embedder
from kornia.geometry.conversions import quaternion_log_to_exp
from models.geometry_utils import rotate_points_with_quaternions
from pdb import set_trace


class MLP(nn.Module):
    def __init__(self, c_in, c_out, c_hiddens, act=nn.LeakyReLU, bn=nn.BatchNorm1d, zero_init=False):
        super().__init__()
        layers = []
        d_in = c_in
        for d_out in c_hiddens:
            layers.append(nn.Linear(d_in, d_out))  # nn.Conv1d(d_in, d_out, 1, 1, 0)
            if bn is not None:
                layers.append(bn(d_out))
            layers.append(act())
            d_in = d_out
        layers.append(nn.Linear(d_in, c_out))  # nn.Conv1d(d_in, c_out, 1, 1, 0)
        if zero_init:
            nn.init.constant_(layers[-1].bias, 0.0)
            nn.init.constant_(layers[-1].weight, 0.0)
        self.mlp = nn.Sequential(*layers)
        self.c_out = c_out

    def forward(self, x):
        return self.mlp(x)


class Shift(nn.Module):
    def __init__(self, shift) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x):
        return x + self.shift


class BaseProjectionLayer(nn.Module):
    @property
    def proj_dims(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


class ProjectionLayer(BaseProjectionLayer):
    def __init__(self, input_dims, proj_dims):
        super().__init__()
        self._proj_dims = proj_dims

        self.proj = nn.Sequential(
            nn.Linear(input_dims, 2 * proj_dims), nn.ReLU(), nn.Linear(2 * proj_dims, proj_dims)
        )

    @property
    def proj_dims(self):
        return self._proj_dims

    def forward(self, x):
        return self.proj(x)


class CouplingLayer(nn.Module):
    def __init__(self, map_s, map_t, projection, mask):
        super().__init__()
        self.map_s = map_s
        self.map_t = map_t
        self.projection = projection
        self.register_buffer("mask", mask)  # 1,1,1,3 -> 1,3

    def forward(self, F, y, alpha_ratio):
        y1 = y * self.mask

        F_y1 = torch.cat([F, self.projection(y1, alpha_ratio)], dim=-1)
        s = self.map_s(F_y1)
        t = self.map_t(F_y1)

        x = y1 + (1 - self.mask) * ((y - t) * torch.exp(-s))
        ldj = (-s).sum(-1)

        return x, ldj

    def inverse(self, F, x, alpha_ratio):
        x1 = x * self.mask

        F_x1 = torch.cat([F, self.projection(x1, alpha_ratio)], dim=-1)
        s = self.map_s(F_x1)
        t = self.map_t(F_x1)

        y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
        ldj = s.sum(-1)

        return y, ldj


class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, lnorm=False):
        super(DenseLayer, self).__init__()

        if lnorm:
            self.linear_layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                              nn.LayerNorm(out_dim))
        else:
            self.linear_layer = nn.Linear(in_dim, out_dim)

        if activation is None:
            self.activation = nn.ReLU()
        else:
            self.activation = activation

    def forward(self, x):
        out = self.linear_layer(x)
        out = self.activation(out)
        return out


def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


def euler2rot_inv(euler_angle):
    batch_size = euler_angle.shape[0]
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))


def euler2rot_2d(euler_angle):
    # (B, 1) -> (B, 2, 2)
    theta = euler_angle.reshape(-1, 1, 1)
    rot = torch.cat((
        torch.cat((theta.cos(), theta.sin()), 1),
        torch.cat((-theta.sin(), theta.cos()), 1),
    ), 2)

    return rot


def euler2rot_2dinv(euler_angle):
    # (B, 1) -> (B, 2, 2)
    theta = euler_angle.reshape(-1, 1, 1)
    rot = torch.cat((
        torch.cat((theta.cos(), -theta.sin()), 1),
        torch.cat((theta.sin(), theta.cos()), 1),
    ), 2)

    return rot


def quaternions_to_rotation_matrices(quaternions):
    """
    Arguments:
    ---------
        quaternions: Tensor with size ...x4, where ... denotes any shape of
                     quaternions to be translated to rotation matrices
    Returns:
    -------
        rotation_matrices: Tensor with size ...x3x3, that contains the computed
                           rotation matrices
    """
    # Allocate memory for a Tensor of size ...x3x3 that will hold the rotation
    # matrix along the x-axis
    shape = quaternions.shape[:-1] + (3, 3)
    R = quaternions.new_zeros(shape)

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[..., 1] ** 2
    yy = quaternions[..., 2] ** 2
    zz = quaternions[..., 3] ** 2
    ww = quaternions[..., 0] ** 2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = torch.zeros_like(n)
    s[n != 0] = 2 / n[n != 0]

    xy = s[..., 0] * quaternions[..., 1] * quaternions[..., 2]
    xz = s[..., 0] * quaternions[..., 1] * quaternions[..., 3]
    yz = s[..., 0] * quaternions[..., 2] * quaternions[..., 3]
    xw = s[..., 0] * quaternions[..., 1] * quaternions[..., 0]
    yw = s[..., 0] * quaternions[..., 2] * quaternions[..., 0]
    zw = s[..., 0] * quaternions[..., 3] * quaternions[..., 0]

    xx = s[..., 0] * xx
    yy = s[..., 0] * yy
    zz = s[..., 0] * zz

    R[..., 0, 0] = 1 - yy - zz
    R[..., 0, 1] = xy - zw
    R[..., 0, 2] = xz + yw

    R[..., 1, 0] = xy + zw
    R[..., 1, 1] = 1 - xx - zz
    R[..., 1, 2] = yz - xw

    R[..., 2, 0] = xz - yw
    R[..., 2, 1] = yz + xw
    R[..., 2, 2] = 1 - xx - yy

    return R


# Deform
class DeformNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_in,
                 d_out_1,
                 d_out_2,
                 n_blocks,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 weight_norm=True):
        super(DeformNetwork, self).__init__()

        self.n_blocks = n_blocks
        self.skip_in = skip_in

        # part a
        # xy -> z
        ori_in = d_in - 1
        dims_in = ori_in
        dims = [dims_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out_1]

        self.embed_fn_1 = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=dims_in)
            self.embed_fn_1 = embed_fn
            dims_in = input_ch
            dims[0] = input_ch + d_feature

        self.num_layers_a = len(dims)
        for i_b in range(self.n_blocks):
            for l in range(0, self.num_layers_a - 1):
                if l + 1 in self.skip_in:
                    out_dim = dims[l + 1] - dims_in
                else:
                    out_dim = dims[l + 1]

                lin = nn.Linear(dims[l], out_dim)

                if l == self.num_layers_a - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight, 0.0)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :ori_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, ori_in:], 0.0)
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :-(dims_in - ori_in)], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_in - ori_in):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

                if weight_norm and l < self.num_layers_a - 2:
                    lin = nn.utils.weight_norm(lin)

                setattr(self, "lin" + str(i_b) + "_a_" + str(l), lin)

        # part b
        # z -> xy
        ori_in = 1
        dims_in = ori_in
        dims = [dims_in + d_feature] + [d_hidden] + [d_out_2]

        self.embed_fn_2 = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=dims_in)
            self.embed_fn_2 = embed_fn
            dims_in = input_ch
            dims[0] = input_ch + d_feature

        self.num_layers_b = len(dims)
        for i_b in range(self.n_blocks):
            for l in range(0, self.num_layers_b - 1):
                if l + 1 in self.skip_in:
                    out_dim = dims[l + 1] - dims_in
                else:
                    out_dim = dims[l + 1]

                lin = nn.Linear(dims[l], out_dim)

                if l == self.num_layers_b - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight, 0.0)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :ori_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, ori_in:], 0.0)
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :-(dims_in - ori_in)], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_in - ori_in):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

                if weight_norm and l < self.num_layers_b - 2:
                    lin = nn.utils.weight_norm(lin)

                setattr(self, "lin" + str(i_b) + "_b_" + str(l), lin)

        # latent code
        for i_b in range(self.n_blocks):
            lin = nn.Linear(d_feature, d_feature)
            torch.nn.init.constant_(lin.bias, 0.0)
            torch.nn.init.constant_(lin.weight, 0.0)
            setattr(self, "lin" + str(i_b) + "_c", lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, deformation_code, input_pts, alpha_ratio, log_quan=None):
        batch_size = input_pts.shape[0]
        x = input_pts
        num_rays, lat_dim = deformation_code.shape
        samples = batch_size // num_rays
        for i_b in range(self.n_blocks):
            form = (i_b // 3) % 2
            mode = i_b % 3

            lin = getattr(self, "lin" + str(i_b) + "_c")
            # interesting! they do a linear residual mapping then concatenate deformation code why??
            deform_code_ib = lin(deformation_code) + deformation_code
            deform_code_ib = deform_code_ib.repeat(samples, 1).reshape(-1, lat_dim)
            #deform_code_ib = deform_code_ib.repeat(batch_size, 1)
            # part a
            if form == 0:
                # zyx
                if mode == 0:
                    x_focus = x[:, [2]]
                    x_other = x[:, [0, 1]]
                elif mode == 1:
                    x_focus = x[:, [1]]
                    x_other = x[:, [0, 2]]
                else:
                    x_focus = x[:, [0]]
                    x_other = x[:, [1, 2]]
            else:
                # xyz
                if mode == 0:
                    x_focus = x[:, [0]]
                    x_other = x[:, [1, 2]]
                elif mode == 1:
                    x_focus = x[:, [1]]
                    x_other = x[:, [0, 2]]
                else:
                    x_focus = x[:, [2]]
                    x_other = x[:, [0, 1]]
            x_ori = x_other  # xy
            if self.embed_fn_1 is not None:
                # Anneal
                x_other = self.embed_fn_1(x_other, alpha_ratio)
            x_other = torch.cat([x_other, deform_code_ib], dim=-1)
            x = x_other
            for l in range(0, self.num_layers_a - 1):
                lin = getattr(self, "lin" + str(i_b) + "_a_" + str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_other], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_a - 2:
                    x = self.activation(x)

            x_focus = x_focus - x

            # part b
            x_focus_ori = x_focus  # z'
            if self.embed_fn_2 is not None:
                # Anneal
                x_focus = self.embed_fn_2(x_focus, alpha_ratio)
            x_focus = torch.cat([x_focus, deform_code_ib], dim=-1)
            x = x_focus
            for l in range(0, self.num_layers_b - 1):
                lin = getattr(self, "lin" + str(i_b) + "_b_" + str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_focus], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_b - 2:
                    x = self.activation(x)

            rot_2d = euler2rot_2dinv(x[:, [0]])
            trans_2d = x[:, 1:]
            x_other = torch.bmm(rot_2d, (x_ori - trans_2d)[..., None]).squeeze(-1)
            if form == 0:
                if mode == 0:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_other[:, [0]], x_focus_ori, x_other[:, [1]]], dim=-1)
                else:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)
            else:
                if mode == 0:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_other[:, [0]], x_focus_ori, x_other[:, [1]]], dim=-1)
                else:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)

        return x

    def inverse(self, deformation_code, input_pts, alpha_ratio, log_quan=None):
        batch_size = input_pts.shape[0]
        x = input_pts
        num_rays, lat_dim = deformation_code.shape
        samples = batch_size // num_rays
        for i_b in range(self.n_blocks):
            i_b = self.n_blocks - 1 - i_b  # inverse
            form = (i_b // 3) % 2
            mode = i_b % 3

            lin = getattr(self, "lin" + str(i_b) + "_c")
            deform_code_ib = lin(deformation_code) + deformation_code
            deform_code_ib = deform_code_ib.repeat(samples, 1).reshape(-1, lat_dim)
            #deform_code_ib = deform_code_ib.repeat(batch_size, 1)
            # part b
            if form == 0:
                # axis: z -> y -> x
                if mode == 0:
                    x_focus = x[:, [0, 1]]
                    x_other = x[:, [2]]
                elif mode == 1:
                    x_focus = x[:, [0, 2]]
                    x_other = x[:, [1]]
                else:
                    x_focus = x[:, [1, 2]]
                    x_other = x[:, [0]]
            else:
                # axis: x -> y -> z
                if mode == 0:
                    x_focus = x[:, [1, 2]]
                    x_other = x[:, [0]]
                elif mode == 1:
                    x_focus = x[:, [0, 2]]
                    x_other = x[:, [1]]
                else:
                    x_focus = x[:, [0, 1]]
                    x_other = x[:, [2]]
            x_ori = x_other  # z'
            if self.embed_fn_2 is not None:
                # Anneal
                x_other = self.embed_fn_2(x_other, alpha_ratio)
            x_other = torch.cat([x_other, deform_code_ib], dim=-1)
            x = x_other
            for l in range(0, self.num_layers_b - 1):
                lin = getattr(self, "lin" + str(i_b) + "_b_" + str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_other], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_b - 2:
                    x = self.activation(x)

            rot_2d = euler2rot_2d(x[:, [0]])
            trans_2d = x[:, 1:]
            x_focus = torch.bmm(rot_2d, x_focus[..., None]).squeeze(-1) + trans_2d

            # part a
            x_focus_ori = x_focus  # xy
            if self.embed_fn_1 is not None:
                # Anneal
                x_focus = self.embed_fn_1(x_focus, alpha_ratio)
            x_focus = torch.cat([x_focus, deform_code_ib], dim=-1)
            x = x_focus
            for l in range(0, self.num_layers_a - 1):
                lin = getattr(self, "lin" + str(i_b) + "_a_" + str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_focus], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_a - 2:
                    x = self.activation(x)

            x_other = x_ori + x
            if form == 0:
                if mode == 0:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_focus_ori[:, [0]], x_other, x_focus_ori[:, [1]]], dim=-1)
                else:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)
            else:
                if mode == 0:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_focus_ori[:, [0]], x_other, x_focus_ori[:, [1]]], dim=-1)
                else:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)

        return x


class SODeformaNet(nn.Module):
    def __init__(self,
                 d_feature,  # input_lat_ch
                 d_in,  #
                 d_out_1,
                 d_out_2,
                 n_blocks,  # [D]
                 d_hidden,  # W
                 n_layers,  # D
                 skip_in=(4,),
                 multires=0,
                 weight_norm=True
                 ):
        super(SODeformaNet, self).__init__()
        dims_in = d_in
        W = d_hidden
        input_lat_ch = d_feature
        D = n_layers * n_blocks * 2
        self.embed_fn, input_ch = get_embedder(multires, input_dims=dims_in)
        self.x_pe_dim = input_ch
        self.W = W
        self.D = D
        self.skips = skip_in
        layers = []
        input_ch = input_ch + input_lat_ch

        for l in range(D + 1):
            if l == D:
                out_dim = 3
            else:
                out_dim = W

            if l == 0:
                in_dim = input_ch
            elif l in self.skips:
                in_dim = input_ch + W
            else:
                in_dim = W

            if l != D:
                layer = DenseLayer(in_dim, out_dim)
            else:
                layer = nn.Linear(in_dim, out_dim)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, t, x, alpha_ratio, log_quan=None, return_pe=False):
        batch_size = x.shape[0]
        t = t.repeat(batch_size, 1)
        if log_quan is not None:
            log_quan = log_quan.repeat(batch_size, 1)
            x = self.apply_quan(x, log_quan)
        x_pe = self.embed_fn(x, alpha_ratio)
        # x_pe = self.embed_fn(x)
        # delta_x = torch.zeros_like(x[..., :3])
        # first_f_idx = ~(t == 0).all(-1)
        feat = torch.cat([x_pe, t], dim=-1)
        h = feat
        for i in range(self.D + 1):
            if i in self.skips:
                h = torch.cat([h, feat], dim=-1)
            h = self.layers[i](h)
        # delta_x[first_f_idx] = h[first_f_idx].contiguous()
        delta_x = h
        x_warped = x + delta_x
        # x_warped = self.apply_quan(x_warped, log_quan)
        if return_pe:  # return x_pe
            return x_warped, x_pe
        else:
            return x_warped

    def inverse(self, t, x, alpha_ratio, log_quan):
        return self.forward(t, x, alpha_ratio, log_quan)

    def apply_quan(self, x, log_quan):

        log_quan_shape = log_quan.shape
        if len(log_quan_shape) == 3:
            log_quan = log_quan.view(-1, log_quan_shape[-1])
            x = x.view(-1, 3)

        # Extract v (rotation), s (pivot point), t (translation)
        v, s, t = log_quan[..., :3], log_quan[..., 3:-3], log_quan[..., -3:]

        # Convert log-quaternion to unit quaternion
        q = quaternion_log_to_exp(v)

        # Points centered around pivot points s
        x_pivot = x - s

        # Apply rotation
        x_rotated = rotate_points_with_quaternions(p=x_pivot, q=q)

        # Transform back to world space by adding s and also add the additional translation
        x_warped = x_rotated + s + t

        if len(log_quan_shape) == 3:
            x_warped = x_warped.view(log_quan_shape[0], log_quan_shape[1], -1)

        return x_warped


class DeformationNetSE3(nn.Module):
    def __init__(self,
                 d_feature,  # input_lat_ch
                 d_in,  #
                 d_out_1,
                 d_out_2,
                 n_blocks,  # [D]
                 d_hidden,  # W
                 n_layers,  # D
                 skip_in=(4,),
                 multires=0,
                 weight_norm=True,
                 ):
        super(DeformationNetSE3, self).__init__()
        dims_in = d_in
        W = d_hidden
        input_lat_ch = d_feature
        D = n_layers * n_blocks * 2
        self.embed_fn, input_ch = get_embedder(multires, input_dims=dims_in)
        self.x_pe_dim = input_ch
        self.W = W
        self.D = D
        self.skips = skip_in
        layers = []
        input_ch = input_ch + input_lat_ch

        for l in range(D+1):
            if l == D:
                out_dim = 9
            else:
                out_dim = W

            if l == 0:
                in_dim = input_ch
            elif l in self.skips:
                in_dim = input_ch + W
            else:
                in_dim = W

            if l != D:
                layer = DenseLayer(in_dim, out_dim)
            else:
                layer = nn.Linear(in_dim, out_dim)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, t, x, alpha_ratio=0., log_quan=None, return_pe=False):
        batch_size = x.shape[0]
        t = t.repeat(batch_size, 1)
        if log_quan is not None:
            log_quan = log_quan.repeat(batch_size, 1)
            x = self.apply_quan(x, log_quan)
        x_pe = self.embed_fn(x, alpha_ratio)
        # x_pe = self.embed_fn(x)
        # delta_x = torch.zeros_like(x[..., :3])
        # first_f_idx = ~(t == 0).all(-1)
        feat = torch.cat([x_pe, t], dim=-1)
        h = feat
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, feat], dim=-1)
            h = self.layers[i](h)

        #######################################################
        # Apply SE(3) transformation to input point xyz
        #######################################################
        h_shape = h.shape
        if len(h_shape) == 3:
            h = h.view(-1, h_shape[-1])
            x = x.view(-1, 3)
        # Extract v (rotation), s (pivot point), t (translation)
        v, s, t = h[..., :3], h[..., 3:-3], h[..., -3:]

        # Convert log-quaternion to unit quaternion
        q = quaternion_log_to_exp(v)

        # Points centered around pivot points s
        x_pivot = x - s

        # Apply rotation
        x_rotated = rotate_points_with_quaternions(p=x_pivot, q=q)

        # Transform back to world space by adding s and also add the additional translation
        x_warped = x_rotated + s + t

        if len(h_shape) == 3:
            x_warped = x_warped.view(h_shape[0], h_shape[1], -1)

        if return_pe:  # return x_pe
            return x_warped, x_pe
        else:
            return x_warped

    def inverse(self, t, x, alpha_ratio, log_quan):
        return self.forward(t, x, alpha_ratio, log_quan)

    def apply_quan(self, x, log_quan):

        log_quan_shape = log_quan.shape
        if len(log_quan_shape) == 3:
            log_quan = log_quan.view(-1, log_quan_shape[-1])
            x = x.view(-1, 3)

        # Extract v (rotation), s (pivot point), t (translation)
        v, s, t = log_quan[..., :3], log_quan[..., 3:-3], log_quan[..., -3:]

        # Convert log-quaternion to unit quaternion
        q = quaternion_log_to_exp(v)

        # Points centered around pivot points s
        x_pivot = x - s

        # Apply rotation
        x_rotated = rotate_points_with_quaternions(p=x_pivot, q=q)

        # Transform back to world space by adding s and also add the additional translation
        x_warped = x_rotated + s + t

        if len(log_quan_shape) == 3:
            x_warped = x_warped.view(log_quan_shape[0], log_quan_shape[1], -1)

        return x_warped


# Deform
class TopoNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 weight_norm=True,
                 use_topo=True):
        super(TopoNetwork, self).__init__()

        dims_in = d_in
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None
        self.use_topo = use_topo

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims_in = input_ch
            dims[0] = input_ch + d_feature

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims_in
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=0.0, std=1e-5)
                torch.nn.init.constant_(lin.bias, bias)
            elif multires > 0 and l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, d_in:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :d_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            elif multires > 0 and l in self.skip_in:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                torch.nn.init.constant_(lin.weight[:, -(dims_in - d_in):], 0.0)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, deformation_code, input_pts, alpha_ratio):
        if self.use_topo:
            if self.embed_fn_fine is not None:
                # Anneal
                input_pts = self.embed_fn_fine(input_pts, alpha_ratio)
            #x = torch.cat([input_pts, deformation_code.repeat(input_pts.shape[0], 1)], dim=-1)
            num_rays, lat_dim = deformation_code.shape
            samples = input_pts.shape[0] // num_rays
            deformation_code = deformation_code.repeat(samples, 1).reshape(-1, lat_dim)
            x = torch.cat([input_pts, deformation_code], dim=-1)
            for l in range(0, self.num_layers - 1):
                lin = getattr(self, "lin" + str(l))

                if l in self.skip_in:
                    x = torch.cat([x, input_pts], 1) / np.sqrt(2)

                x = lin(x)

                if l < self.num_layers - 2:
                    x = self.activation(x)

            return x
        else:
            return None


# Deform
class AppearanceNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_global_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature + d_global_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, global_feature, points, normals, view_dirs, feature_vectors, alpha_ratio):
        if self.embedview_fn is not None:
            # Anneal
            view_dirs = self.embedview_fn(view_dirs, alpha_ratio)

        rendering_input = None

        global_feature = global_feature.repeat(points.shape[0], 1)
        if len(feature_vectors.shape) == 3:
            feature_vectors = feature_vectors.squeeze(0)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors, global_feature], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors, global_feature], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors, global_feature], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in_1,
                 d_in_2,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 multires_topo=0,  # TODO: check if they also use that. Ans: they use 1
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in_1 + d_in_2] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None
        self.embed_amb_fn = None

        input_ch_1 = d_in_1
        input_ch_2 = d_in_2
        if multires > 0:
            embed_fn, input_ch_1 = get_embedder(multires, input_dims=d_in_1)
            self.embed_fn_fine = embed_fn
            dims[0] += (input_ch_1 - d_in_1)
        if multires_topo > 0:
            embed_amb_fn, input_ch_2 = get_embedder(multires_topo, input_dims=d_in_2)
            self.embed_amb_fn = embed_amb_fn
            dims[0] += (input_ch_2 - d_in_2)

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, d_in_1:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :d_in_1], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    if multires > 0:
                        torch.nn.init.constant_(lin.weight[:, -(dims[0] - d_in_1):-input_ch_2], 0.0)
                    if multires_topo > 0:
                        torch.nn.init.constant_(lin.weight[:, -(input_ch_2 - d_in_2):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, input_pts, topo_coord, alpha_ratio):
        input_pts = input_pts * self.scale
        if self.embed_fn_fine is not None:
            # Anneal
            input_pts = self.embed_fn_fine(input_pts, alpha_ratio)
        if self.embed_amb_fn is not None and topo_coord is not None:
            # Anneal
            topo_coord = self.embed_amb_fn(topo_coord, alpha_ratio)
        if topo_coord is not None:
            inputs = torch.cat([input_pts, topo_coord], dim=-1)
        else:
            inputs = input_pts
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    # Anneal
    def sdf(self, x, topo_coord, alpha_ratio):
        return self.forward(x, topo_coord, alpha_ratio)[:, :1]

    def sdf_hidden_appearance(self, x, topo_coord, alpha_ratio):
        return self.forward(x, topo_coord, alpha_ratio)

    def gradient(self, x, topo_coord, alpha_ratio):
        x.requires_grad_(True)
        y = self.sdf(x, topo_coord, alpha_ratio)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients


# This implementation is based upon IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None
        if len(feature_vectors.shape) == 3:
            feature_vectors = feature_vectors.squeeze(0)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)


class GeometryNet(nn.Module):
    def __init__(self,
                 d_in_1,  # input_feat_dim
                 d_in_2,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 multires_topo=0,  # TODO: check if they also use that. Ans: they use 1
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 concat_qp=False):
        super(GeometryNet, self).__init__()
        W = d_hidden
        input_feat_dim = d_in_1 + d_in_2
        D = n_layers
        skips = skip_in
        n_freq = multires
        self.embed_fn, input_ch = get_embedder(n_freq, input_dims=input_feat_dim + concat_qp * 3)
        self.W = W
        self.D = D
        self.skips = skips
        self.n_freq = n_freq
        layers = []

        for l in range(D + 1):
            if l == D:
                out_dim = 1
            else:
                out_dim = W

            if l == 0:
                in_dim = input_ch
            elif l in skips:
                in_dim = input_ch + W
            else:
                in_dim = W

            if l != D:
                layer = DenseLayer(in_dim, out_dim)
            else:
                layer = nn.Linear(in_dim, out_dim)

            if weight_norm and l == D:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, feat, t=None):
        feat_input = self.embed_fn(feat, self.n_freq)
        h = feat_input

        for i in range(self.D + 1):
            if i in self.skips:
                h = torch.cat([h, feat_input], dim=-1)
            h = self.layers[i](h)

        return h


class RadianceNet(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True,
                 use_view_dirs=True,
                 ):
        super(RadianceNet, self).__init__()
        W = d_hidden
        D = n_layers
        skips = []
        n_freq = multires_view
        input_feat_dim = d_in
        if use_view_dirs:
            self.embed_fn, input_ch = get_embedder(n_freq, input_dims=3)
            input_ch += input_feat_dim
        else:
            self.embed_fn = None
            input_ch = input_feat_dim

        self.use_view_dirs = use_view_dirs
        self.W = W
        self.D = D
        self.skips = skips
        self.n_freq = n_freq
        layers = []

        for l in range(D + 1):
            if l == D:
                out_dim = 3
            else:
                out_dim = W

            if l == 0:
                in_dim = input_ch
            elif l in self.skips:
                in_dim = input_ch + W
            else:
                in_dim = W

            if l != D:
                layer = DenseLayer(in_dim, out_dim)
            else:
                layer = nn.Linear(in_dim, out_dim)

            if weight_norm and l == D:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, feat, viewdirs=None, t=None):
        if self.use_view_dirs and viewdirs is not None:
            viewdirs = self.embed_fn(viewdirs, self.n_freq)
            if len(feat.shape) == 3:
                viewdirs = viewdirs.unsqueeze(0)
            feat = torch.cat([feat, viewdirs], dim=-1)
        h = feat

        for i in range(self.D + 1):
            if i in self.skips:
                h = torch.cat([h, feat], dim=-1)
            h = self.layers[i](h)

        return h


class GeometryNetLat(nn.Module):
    def __init__(self,
                 d_in_1,  # input_feat_dim
                 d_in_2,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 multires_topo=0,  # TODO: check if they also use that. Ans: they use 1
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 concat_qp=False,
                 ):
        super(GeometryNetLat, self).__init__()
        W = d_hidden
        input_feat_dim = d_in_1
        input_lat_ch = d_in_2
        D = n_layers
        skips = skip_in
        n_freq = multires
        self.embed_fn, input_ch = get_embedder(n_freq, input_dim=input_feat_dim + concat_qp * 3)
        self.W = W
        self.D = D
        self.skips = skips
        layers = []
        input_ch = input_ch + input_lat_ch

        for l in range(D + 1):
            if l == D:
                out_dim = 1
            else:
                out_dim = W

            if l == 0:
                in_dim = input_ch
            elif l in skips:
                in_dim = input_ch + W
            else:
                in_dim = W

            if l != D:
                layer = DenseLayer(in_dim, out_dim)
            else:
                layer = nn.Linear(in_dim, out_dim)

            if weight_norm and l == D:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, feat, t):
        feat_input = self.embed_fn(feat)
        feat_input = torch.cat([feat_input, t], dim=-1)
        h = feat_input

        for i in range(self.D + 1):
            if i in self.skips:
                h = torch.cat([h, feat_input], dim=-1)
            h = self.layers[i](h)

        return h


class RadianceNetLat(nn.Module):
    def __init__(self,
                 d_feature,
                 mode,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 weight_norm=True,
                 multires_view=0,
                 squeeze_out=True,
                 use_view_dirs=True,
                 ):
        super(RadianceNetLat, self).__init__()
        W = d_hidden
        D = n_layers
        skips = []
        n_freq = multires_view
        input_feat_dim = d_in
        input_lat_ch = d_feature
        if use_view_dirs:
            self.embed_fn, input_ch = get_embedder(n_freq, input_dim=3)
            input_ch += input_feat_dim
        else:
            self.embed_fn = None
            input_ch = input_feat_dim

        self.use_view_dirs = use_view_dirs
        self.W = W
        self.D = D
        self.skips = skips
        layers = []
        input_ch = input_ch + input_lat_ch

        for l in range(D + 1):
            if l == D:
                out_dim = 3
            else:
                out_dim = W

            if l == 0:
                in_dim = input_ch
            elif l in self.skips:
                in_dim = input_ch + W
            else:
                in_dim = W

            if l != D:
                layer = DenseLayer(in_dim, out_dim)
            else:
                layer = nn.Linear(in_dim, out_dim)

            if weight_norm and l ==D:
                layer = nn.utils.weight_norm(layer)

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(self, feat, t, viewdirs=None):
        if self.use_view_dirs and viewdirs is not None:
            viewdirs = self.embed_fn(viewdirs)
            feat = torch.cat([feat, viewdirs], dim=-1)
        feat = torch.cat([feat, t], dim=-1)
        h = feat

        for i in range(self.D + 1):
            if i in self.skips:
                h = torch.cat([h, feat], dim=-1)
            h = self.layers[i](h)

        return h
