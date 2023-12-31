import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import mcubes
from models.rend_utils import qp_to_sdf, neus_weights, get_sdf_loss
from models.losses import compute_akap_loss, compute_elastic_loss, general_loss_with_squared_residual, get_normal_loss
from pdb import set_trace


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 128  # 64. Change it when memory is insufficient!
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('Threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is built upon NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, sdf, beta):
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


# Deform
class DeformGoSRenderer:
    def __init__(self,
                 report_freq,
                 deform_network,
                 ambient_network,
                 sdf_network,
                 deviation_network,
                 color_network,
                 feature_grid,
                 volume_origin,
                 volume_dim,
                 rgb_dim,
                 begin_n_samples,
                 end_n_samples,
                 important_begin_iter,
                 n_importance,
                 up_sample_steps,
                 perturb,
                 ndr_color_net=False):
        self.dtype = torch.get_default_dtype()
        # Deform
        self.deform_network = deform_network
        self.ambient_network = ambient_network
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.begin_n_samples = begin_n_samples
        self.end_n_samples = end_n_samples
        self.n_samples = self.begin_n_samples
        self.important_begin_iter = important_begin_iter
        self.n_importance = n_importance
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.report_freq = report_freq
        self.ndr_color = ndr_color_net
        # Grid
        self.feature_grid = feature_grid
        self.volume_origin = volume_origin
        self.volume_dim = volume_dim
        self.rgb_dim = rgb_dim

        self.laplace_density = LaplaceDensity(0.01)

    def update_samples_num(self, iter_step, alpha_ratio=0.):
        if iter_step >= self.important_begin_iter:
            self.n_samples = int(self.begin_n_samples * (1 - alpha_ratio) + self.end_n_samples * alpha_ratio)

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, deform_code, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False,
                   alpha_ratio=0.0, volume_origin=None, volume_dim=None, feature_grid=None, log_qua=None, iter_step=0):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
            pts = pts.reshape(-1, 3)
            # Deform
            pts_canonical = self.deform_network(deform_code, pts, alpha_ratio, log_qua)
            ambient_coord = self.ambient_network(deform_code, pts, alpha_ratio, iter_step=iter_step)
            new_sdf, *_ = qp_to_sdf(pts_canonical, volume_origin, volume_dim, feature_grid,
                                    self.sdf_network, hyper_embed=ambient_coord, rgb_dim=self.rgb_dim)
            if len(sdf.shape) == 2:
                sdf = sdf.reshape(-1, 1)
            sdf = torch.cat([sdf, new_sdf], dim=0)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[index].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    deform_code,
                    appearance_code,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    deform_network,
                    ambient_network,
                    sdf_network,
                    deviation_network,
                    color_network,
                    cos_anneal_ratio=0.0,
                    alpha_ratio=0.,
                    rays_l=None,
                    truncation=0.08,
                    back_truncation=0.01,
                    log_qua=None,
                    gt_normals=None,
                    iter_step=0,):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs_o = rays_d[:, None, :].expand(pts.shape)  # view in observation space

        pts = pts.reshape(-1, 3)
        dirs_o = dirs_o.reshape(-1, 3)

        # Deform
        # observation space -> canonical space
        pts_canonical = deform_network(deform_code, pts, alpha_ratio, log_qua)
        ambient_coord = ambient_network(deform_code, pts, alpha_ratio, iter_step=iter_step)
        # TODO: without alpha_ratio for now
        sdf, feature_vector, _ = qp_to_sdf(pts_canonical, self.volume_origin, self.volume_dim, self.feature_grid,
                                           sdf_network, hyper_embed=ambient_coord, rgb_dim=self.rgb_dim)

        # Deform, gradients in observation space
        def gradient(deform_network=None, ambient_network=None, sdf_network=None, deform_code=None, x=None,
                     alpha_ratio=None, volume_origin=None, volume_dim=None, feature_grid=None, rgb_dim=6, log_qua=None):
            x.requires_grad_(True)
            x_c = deform_network(deform_code, x, alpha_ratio, log_qua)
            amb_coord = ambient_network(deform_code, x, alpha_ratio, iter_step=iter_step)
            y, *_ = qp_to_sdf(x_c, volume_origin, volume_dim, feature_grid, sdf_network, hyper_embed=amb_coord,
                              rgb_dim=rgb_dim)

            # gradient on observation
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradient_o = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

            # Jacobian on pts
            y_0 = x_c[:, 0]
            y_1 = x_c[:, 1]
            y_2 = x_c[:, 2]
            d_output = torch.ones_like(y_0, requires_grad=False, device=y_0.device)
            grad_0 = torch.autograd.grad(
                outputs=y_0,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            grad_1 = torch.autograd.grad(
                outputs=y_1,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            grad_2 = torch.autograd.grad(
                outputs=y_2,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            gradient_pts = torch.cat([grad_0, grad_1, grad_2], dim=1)  # (batch_size, dim_out, dim_in)
            return gradient_o, gradient_pts

        # Deform
        # observation space -> canonical space
        gradients_o, pts_jacobian = gradient(deform_network, ambient_network, sdf_network, deform_code, pts,
                                             alpha_ratio, volume_origin=self.volume_origin, volume_dim=self.volume_dim,
                                             feature_grid=self.feature_grid, log_qua=log_qua, rgb_dim=self.rgb_dim)
        dirs_c = torch.bmm(pts_jacobian, dirs_o.unsqueeze(-1)).squeeze(-1)  # view in observation space
        dirs_c = dirs_c / torch.linalg.norm(dirs_c, ord=2, dim=-1, keepdim=True)
        if self.ndr_color:
            sampled_color = color_network(appearance_code, pts_canonical, gradients_o, \
                                      dirs_c, feature_vector, alpha_ratio).reshape(batch_size, n_samples, 3)
        else:
            sampled_color = torch.sigmoid(color_network(feature_vector, dirs_c)).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs_o * gradients_o).sum(-1, keepdim=True)  # observation

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).to(self.dtype).detach()
        relax_inside_sphere = (pts_norm < 1.2).to(self.dtype).detach()

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)
        weights_ours, _, _ = neus_weights(sdf, dists, inv_s, z_vals=None, view_dirs=dirs_o, grads=gradients_o,
                                          cos_val=true_cos, batch_size=batch_size, n_samples=n_samples)
        # depth map
        depth_map = torch.sum(weights * mid_z_vals, -1, keepdim=True)
        depth_map_ours = torch.sum(weights_ours * mid_z_vals, -1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        color_ours = (sampled_color * weights_ours[:, :, None]).sum(dim=1)

        # Eikonal loss, observation + canonical
        gradient_o_error = (torch.linalg.norm(gradients_o.reshape(batch_size, n_samples, 3), ord=2,
                                              dim=-1) - 1.0) ** 2
        relax_inside_sphere_sum = relax_inside_sphere.sum() + 1e-5
        #gradient_o_error = (relax_inside_sphere * gradient_o_error).sum() / relax_inside_sphere_sum
        gradient_o_error = gradient_o_error.mean()
        if rays_l is not None:
            sdf_pred = sdf.reshape(batch_size, n_samples)
            fs_loss, sdf_loss = get_sdf_loss(mid_z_vals, rays_l, sdf_pred, truncation, back_truncation, inside_sphere)
        else:
            fs_loss, sdf_loss = 0., 0.

        akap_loss = compute_akap_loss(pts_jacobian)
        elastic_loss, residual = 0.0, 0.0 #compute_elastic_loss(pts_jacobian)

        pred_normals = gradients_o.reshape(batch_size, n_samples, 3) * weights[:, :n_samples, None]
        pred_normals = pred_normals * inside_sphere[..., None]
        pred_normals = pred_normals.sum(dim=1)

        #pred_normals = torch.nn.functional.normalize(pred_normals, p=2, dim=-1)
        if gt_normals is not None:
            normal_l1, normal_cos = get_normal_loss(pred_normals, gt_normals)
        else:
            normal_l1, normal_cos = 0.0, 0.0

        return {
            'pts': pts.reshape(batch_size, n_samples, 3),
            'pts_canonical': pts_canonical.reshape(batch_size, n_samples, 3),
            'relax_inside_sphere': relax_inside_sphere,
            'color': color,
            'color_ours': color_ours,
            'sdf': sdf,
            'dists': dists,
            'gradients_o': gradients_o.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'weights_sum': weights_sum,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_o_error': gradient_o_error,
            'inside_sphere': inside_sphere,
            'depth_map': depth_map,
            'depth_map_ours': depth_map_ours,
            'fs_loss': fs_loss,
            'sdf_loss': sdf_loss,
            'akap_loss': akap_loss.mean(),
            'elastic_loss': 0.0, #elastic_loss.mean(),
            'normal_l1': normal_l1,
            'normal_cos': normal_cos,
        }

    # TODO: modify this to work with grid
    def render(self, deform_code, appearance_code, rays_o, rays_d, near, far, perturb_overwrite=-1,
               cos_anneal_ratio=0.0, alpha_ratio=0., iter_step=0, rays_l=None, truncation=0.08, back_truncation=0.01,
               log_qua=None, gt_normals=None):
        self.update_samples_num(iter_step, alpha_ratio)
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples  # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        # Up sample
        if iter_step >= self.important_begin_iter and self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                pts = pts.reshape(-1, 3)
                # Deform
                pts_canonical = self.deform_network(deform_code, pts, alpha_ratio, log_qua)
                ambient_coord = self.ambient_network(deform_code, pts, alpha_ratio, iter_step=iter_step)
                sdf, *_ = qp_to_sdf(pts_canonical, self.volume_origin, self.volume_dim, self.feature_grid,
                                    self.sdf_network, hyper_embed=ambient_coord, rgb_dim=self.rgb_dim)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2 ** i)
                    z_vals, sdf = self.cat_z_vals(deform_code,
                                                  rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps),
                                                  alpha_ratio=alpha_ratio,
                                                  volume_origin=self.volume_origin,
                                                  volume_dim=self.volume_dim,
                                                  feature_grid=self.feature_grid,
                                                  log_qua=log_qua,
                                                  iter_step=iter_step)

            n_samples = self.n_samples + self.n_importance

        # Render core
        ret_fine = self.render_core(deform_code,
                                    appearance_code,
                                    rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.deform_network,
                                    self.ambient_network,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    alpha_ratio=alpha_ratio,
                                    rays_l=rays_l,
                                    truncation=truncation,
                                    back_truncation=back_truncation,
                                    log_qua=log_qua,
                                    gt_normals=gt_normals,
                                    iter_step=iter_step)

        weights = ret_fine['weights']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'pts': ret_fine['pts'],
            'pts_canonical': ret_fine['pts_canonical'],
            'relax_inside_sphere': ret_fine['relax_inside_sphere'],
            'color_fine': ret_fine['color'],
            'color_ours': ret_fine['color_ours'],
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': ret_fine['weights_sum'],
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients_o': ret_fine['gradients_o'],
            'weights': weights,
            'gradient_o_error': ret_fine['gradient_o_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'depth_map': ret_fine['depth_map'],
            'fs_loss': ret_fine['fs_loss'],
            'sdf_loss': ret_fine['sdf_loss'],
            'akap_loss': ret_fine['akap_loss'],
            'elastic_loss': ret_fine['elastic_loss'],
            'normal_l1': ret_fine['normal_l1'],
            'normal_cos': ret_fine['normal_cos'],
        }

    # Depth
    # TODO: modify this to work with grid
    def renderondepth(self,
                      deform_code,
                      appearance_code,
                      rays_o,
                      rays_d,
                      rays_s,
                      alpha_ratio=0.,
                      log_qua=None,
                      iter_step=0,):
        pts = rays_o + rays_s  # n_rays, 3

        pts_canonical = self.deform_network(deform_code, pts, alpha_ratio, log_qua)
        ambient_coord = self.ambient_network(deform_code, pts, alpha_ratio, iter_step=iter_step)
        _, feature_vector, _ = qp_to_sdf(pts_canonical, self.volume_origin, self.volume_dim, self.feature_grid,
                                self.sdf_network, hyper_embed=ambient_coord, rgb_dim=self.rgb_dim)

        # Deform, gradients in observation space
        def gradient(deform_network=None, ambient_network=None, sdf_network=None, deform_code=None, x=None,
                     alpha_ratio=None, volume_origin=None, volume_dim=None, feature_grid=None, rgb_dim=6, log_qua=None):
            x.requires_grad_(True)
            x_c = deform_network(deform_code, x, alpha_ratio, log_qua)
            amb_coord = ambient_network(deform_code, x, alpha_ratio, iter_step=iter_step)
            y, *_ = qp_to_sdf(x_c, volume_origin, volume_dim, feature_grid, sdf_network, hyper_embed=amb_coord,
                              rgb_dim=rgb_dim)

            # gradient on observation
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradient_o = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

            ## Jacobian on pts
            y_0 = x_c[:, 0]
            y_1 = x_c[:, 1]
            y_2 = x_c[:, 2]
            d_output = torch.ones_like(y_0, requires_grad=False, device=y_0.device)
            grad_0 = torch.autograd.grad(
                outputs=y_0,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            grad_1 = torch.autograd.grad(
                outputs=y_1,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            grad_2 = torch.autograd.grad(
                outputs=y_2,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0].unsqueeze(1)
            gradient_pts = torch.cat([grad_0, grad_1, grad_2], dim=1)  # (batch_size, dim_out, dim_in)
            return gradient_o, gradient_pts

        gradients_o, pts_jacobian = gradient(self.deform_network, self.ambient_network, self.sdf_network, deform_code,
                                             pts, alpha_ratio, volume_origin=self.volume_origin, volume_dim=self.volume_dim,
                                             feature_grid=self.feature_grid, log_qua=log_qua, rgb_dim=self.rgb_dim)
        dirs_c = torch.bmm(pts_jacobian, rays_d.unsqueeze(-1)).squeeze(-1)  # view in observation space
        dirs_c = dirs_c / torch.linalg.norm(dirs_c, ord=2, dim=-1, keepdim=True)
        if self.ndr_color:
            color = self.color_network(appearance_code, pts_canonical, gradients_o, \
                                   dirs_c, feature_vector, alpha_ratio)
        else:
            color = torch.sigmoid(self.color_network(feature_vector, dirs_c))

        return color, gradients_o

    # Depth
    def errorondepth(self, deform_code, rays_o, rays_d, rays_s, mask, alpha_ratio=0., iter_step=0, log_qua=None,
                     smooth_std=0.01):
        pts = rays_o + rays_s
        pts_canonical = self.deform_network(deform_code, pts, alpha_ratio, log_qua)
        ambient_coord = self.ambient_network(deform_code, pts, alpha_ratio, iter_step=iter_step)
        if iter_step % self.report_freq == 0:
            pts_back = self.deform_network.inverse(deform_code, pts_canonical, alpha_ratio, log_qua)
        sdf, *_ = qp_to_sdf(pts_canonical, self.volume_origin, self.volume_dim, self.feature_grid, self.sdf_network,
                            hyper_embed=ambient_coord, rgb_dim=self.rgb_dim)

        # Deform, gradients in observation space
        def gradient_obs(deform_network=None, ambient_network=None, sdf_network=None, deform_code=None, x=None,
                         alpha_ratio=None, volume_origin=None, volume_dim=None, feature_grid=None, rgb_dim=6, log_qua=None):
            x.requires_grad_(True)
            x_c = deform_network(deform_code, x, alpha_ratio, log_qua)
            amb_coord = ambient_network(deform_code, x, alpha_ratio, iter_step=iter_step)
            # y = sdf_network.sdf(x_c, amb_coord, alpha_ratio)
            y, *_ = qp_to_sdf(x_c, volume_origin, volume_dim, feature_grid, sdf_network, hyper_embed=amb_coord,
                              rgb_dim=rgb_dim)

            # gradient on observation
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradient_o = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]

            return gradient_o

        pts = pts.detach()
        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True)
        # Denoise. not use: out of mask or sphere
        inside_masksphere = (pts_norm < 1.0).to(self.dtype) * mask  # inside_sphere * mask
        sdf = inside_masksphere * sdf
        inside_masksphere_sum = inside_masksphere.sum() + 1e-5
        sdf_error = F.l1_loss(sdf, torch.zeros_like(sdf), reduction='sum') / inside_masksphere_sum
        angle_error = 0.0 #F.l1_loss(relu_cos, torch.zeros_like(relu_cos), reduction='sum') / inside_masksphere_sum

        surface_points_neig = pts + (torch.rand_like(pts) - 0.5) * smooth_std
        pp = torch.cat([pts, surface_points_neig], dim=0)
        surface_grad = gradient_obs(self.deform_network, self.ambient_network, self.sdf_network, deform_code, pp,
                                    alpha_ratio, self.volume_origin, self.volume_dim, self.feature_grid, rgb_dim=self.rgb_dim,
                                    log_qua=log_qua)
        surface_points_normal = torch.nn.functional.normalize(surface_grad, p=2, dim=-1)

        N = surface_points_normal.shape[0] // 2

        diff_norm = torch.norm(surface_points_normal[:N] - surface_points_normal[N:], dim=-1)
        #normal_smoothness_loss = general_loss_with_squared_residual(diff_norm, alpha=-2., scale=3.9).sum()
        normal_smoothness_loss = torch.mean(diff_norm)
        if iter_step % self.report_freq == 0:
            print('Invertibility evaluation: ', torch.abs((pts_back - pts) * inside_masksphere).max().data.item())

        return sdf_error, angle_error, inside_masksphere, normal_smoothness_loss

    def extract_canonical_geometry(self, bound_min, bound_max, resolution, threshold=0.0, alpha_ratio=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: - qp_to_sdf(pts, self.volume_origin, self.volume_dim,
                                                                   self.feature_grid, self.sdf_network,
                                                                   rgb_dim=self.rgb_dim)[0])

    def extract_observation_geometry(self, deform_code, bound_min, bound_max, resolution, threshold=0.0,
                                     alpha_ratio=0.0, log_qua=None):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: - qp_to_sdf(self.deform_network(deform_code, pts, alpha_ratio, log_qua),
                                                                   self.volume_origin, self.volume_dim,
                                                                   self.feature_grid, self.sdf_network,
                                                                   hyper_embed=self.ambient_network(deform_code, pts,
                                                                                                    alpha_ratio,
                                                                                                    iter_step=80000),
                                                                   rgb_dim=self.rgb_dim)[0])

class NeuSRenderer:
    def __init__(self,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 up_sample_steps,
                 perturb):
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts)
        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).to(self.dtype).detach()
        relax_inside_sphere = (pts_norm < 1.2).to(self.dtype).detach()

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples  # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2 ** i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps),
                                                  iter_steps=80000)

            n_samples = self.n_samples + self.n_importance

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))
