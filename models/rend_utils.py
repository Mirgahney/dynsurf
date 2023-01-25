import torch
import torch.nn as nn
import torch.nn.functional as F


def coordinates(voxel_dim, device: torch.device):
    nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))


def get_rays(c2w, intrinsics, H, W, n_rays=5000, mask=None, convention="OpenGL"):
    device = c2w.device
    bs = c2w.shape[0]  # [bs, 4, 4]

    i, j = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing="xy")  # [H, W]

    # TODO: for now all the images within the same batch has the same sampled pixels
    if n_rays > 0:
        if mask is not None:
            valid_bs, valid_hs, valid_ws = torch.nonzero(mask, as_tuple=True)
            n_rays = bs * min(n_rays, len(valid_bs))
            indices = torch.randint(0, len(valid_bs), size=[n_rays], device=device)
            select_bs = valid_bs[indices]
            select_hs = valid_hs[indices]
            select_ws = valid_ws[indices]
        else:
            n_rays = min(n_rays, H*W)
            select_bs = torch.arange(0, bs, dtype=torch.long, device=device).unsqueeze(1).repeat(1, n_rays).view(-1)  # [n_rays * bs]
            select_hs = torch.randint(0, H, size=[n_rays], device=device).repeat(bs)  # [n_rays * bs]
            select_ws = torch.randint(0, W, size=[n_rays], device=device).repeat(bs)  # [n_rays * bs]
    else:  # use entire pixels
        select_bs, select_hs, select_ws = torch.nonzero(mask if mask is not None else torch.ones([bs, H, W]), as_tuple=True)

    i = i[select_hs, select_ws]  # [bs * n_rays]
    j = j[select_hs, select_ws]  # [bs * n_rays]

    assert convention in ["OpenCV", "OpenGL"], "Unknown camera coordinate convention!!!"

    fx = intrinsics[select_bs, 0, 0]  # [bs * n_rays]
    fy = intrinsics[select_bs, 1, 1]
    cx = intrinsics[select_bs, 0, 2]
    cy = intrinsics[select_bs, 1, 2]

    if convention == "OpenCV":  # OpenCV convention
        dirs = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1)  # [bs * n_rays, 3]
    else:
        dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)

    # permute for bmm
    rays_d = torch.bmm(c2w[select_bs, :3, :3], dirs[..., None]).squeeze()  # [bs * n_rays, 3, 3] @ [bs * n_rays, 3, 1]
    rays_o = c2w[select_bs, :3, -1]  # [bs * n_rays, 3]

    return rays_o, rays_d, select_bs.cpu(), select_hs.cpu(), select_ws.cpu()


def batchify(fn, max_chunk=1024*128):
    if max_chunk is None:
        return fn
    def ret(feats):
        chunk = max_chunk // (feats.shape[1] * feats.shape[2])
        return torch.cat([fn(feats[i:i+chunk]) for i in range(0, feats.shape[0], chunk)], dim=0)
    return ret


def render_rays(sdf_decoder,
                rgb_decoder,
                feat_volume,  # feature volume [1, feat_dim, Nx, Ny, Nz]
                volume_origin,  # volume origin, Euclidean coords [3,]
                volume_dim,  # volume dimensions, Euclidean coords [3,]
                voxel_size,  # length of the voxel side, Euclidean distance
                rays_o,
                rays_d,
                truncation=0.10,
                near=0.01,
                far=3.0,
                n_samples=128,
                n_importance=16,
                depth_gt=None,
                inv_s=20.,
                smoothness_std=0.0,
                randomize_samples=True,
                use_view_dirs=False,
                use_normals=False,
                concat_qp_to_rgb=False,
                concat_qp_to_sdf=False,
                surface_wright=0.,
                epoch=0,
                ):

    n_rays = rays_o.shape[0]
    z_vals = torch.linspace(near, far, n_samples).to(rays_o)
    z_vals = z_vals[None, :].repeat(n_rays, 1)  # [n_rays, n_samples]
    sample_dist = (far - near) / n_samples

    if randomize_samples:
        z_vals += torch.rand_like(z_vals) * sample_dist

    n_importance_steps = n_importance // 12
    with torch.no_grad():
        for step in range(n_importance_steps):
            query_points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
            sdf, _ = qp_to_sdf(query_points, volume_origin, volume_dim, feat_volume, sdf_decoder, concat_qp=concat_qp_to_sdf)

            prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
            prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
            mid_sdf = (prev_sdf + next_sdf) * 0.5
            cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

            prev_cos_val = torch.cat([torch.zeros([n_rays, 1], device=z_vals.device), cos_val[:, :-1]], dim=-1)
            cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
            cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
            cos_val = cos_val.clip(-1e3, 0.0)

            dists = next_z_vals - prev_z_vals
            weights, alpha, _ = neus_weights(mid_sdf, dists, 64 * 2 ** step, cos_val=cos_val)
            z_samples = sample_pdf(z_vals, weights, 12, det=True).detach()
            z_vals, sorted_indices = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)

    dists = z_vals[:,1:] - z_vals[:,:-1]
    dists = torch.cat([dists, torch.ones_like(depth_gt).unsqueeze(-1) * sample_dist], dim=1)
    z_vals_mid = z_vals + dists * 0.5
    view_dirs = F.normalize(rays_d, dim=-1)[:, None, :].repeat(1, n_samples + n_importance, 1)
    query_points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_mid[..., :, None]
    query_points = query_points.requires_grad_(True)
    sdf, feat_points = qp_to_sdf(query_points, volume_origin, volume_dim, feat_volume, sdf_decoder, concat_qp=concat_qp_to_sdf)
    grads = compute_grads(sdf, query_points, feat_volume)

    rgb_feat = [from_grid_sample(feat_points[0][:,-8:,...])]

    if use_view_dirs:
        rgb_feat.append(view_dirs)
    if use_normals:
        rgb_feat.append(grads)
    if concat_qp_to_rgb:
        rgb_feat.append(2. * (query_points - volume_origin) / volume_dim - 1.)

    # rgb_feat.append((view_dirs * grads).sum(dim=-1, keepdim=True))

    rgb = torch.sigmoid(rgb_decoder(torch.cat(rgb_feat, dim=-1)))

    weights, z_surf, alpha = neus_weights(sdf, dists, inv_s, z_vals=z_vals_mid, view_dirs=view_dirs,
                                          grads=grads, cos_anneal_ratio=min(epoch/5., 1.))
    weights = weights / weights.sum(dim=-1, keepdim=True)

    rendered_rgb = torch.sum(weights[..., None] * rgb, dim=-2)
    rendered_depth = torch.sum(weights * z_vals_mid, dim=-1)
    # depth_var = torch.sum(weights * torch.square(z_vals_mid - rendered_depth.unsqueeze(-1)), dim=-1)

    # eikonal_loss = torch.square(grads.norm(dim=-1) - 1.)[mask].mean()
    eikonal_mask = ((query_points > volume_origin) & (query_points < (volume_origin + volume_dim))).all(dim=-1)
    # eikonal_weights = sdf[eikonal_mask].detach().abs() + 1e-2
    eikonal_weights = torch.abs(z_vals - depth_gt.unsqueeze(-1))[eikonal_mask] > truncation
    eikonal_loss = (torch.square(grads.norm(dim=-1)[eikonal_mask] - 1.) * eikonal_weights).sum() / eikonal_weights.sum()
    eikonal_loss = eikonal_loss.mean()

    # surface loss
    if surface_wright > 0.:
        z_depth = depth_gt[:, None]  # [n_rays, 1]
        qp_depth = rays_o[:, None, :] + rays_d[:, None, :] * z_depth[..., :, None]  # [n_rays, 1, 3]
        sdf_depth, _ = qp_to_sdf(qp_depth, volume_origin, volume_dim, feat_volume, sdf_decoder, concat_qp=concat_qp_to_sdf)
        surface_loss = F.l1_loss(sdf_depth, torch.zeros_like(sdf_depth))
    else:
        surface_loss = torch.tensor(0.0)

    if depth_gt is not None:
        fs_loss, sdf_loss = get_sdf_loss(z_vals_mid, depth_gt[:, None], sdf, truncation)
    else:
        fs_loss, sdf_loss = 0.0, 0.0

    normal_regularisation_loss = torch.tensor(0., device=z_vals.device)
    if smoothness_std > 0:
        coords = coordinates([feat_volume.volumes[0].shape[2] // 2, feat_volume.volumes[0].shape[3] // 2, feat_volume.volumes[0].shape[4] // 2], z_vals.device).float().t()
        world = (coords * voxel_size * 2 + volume_origin + voxel_size * 2 * torch.rand_like(coords))
        # predicted surface
        z_surf_mask = z_surf[:,0] != z_vals[:,0]
        surf = (rays_o[z_surf_mask,:] + rays_d[z_surf_mask,:] * z_surf[z_surf_mask,:])
        surf_mask = ((surf > volume_origin) & (surf < (volume_origin + volume_dim))).all(dim=-1)
        world = torch.cat([world, surf[surf_mask, :]], dim=0)  # [N, 3]
        # world = get_pts_in_frustum(world, c2w, K, H, W).unsqueeze(0)  # [1, N, 3]
        world = world.unsqueeze(0)

        noise = F.normalize(torch.randn_like(world), dim=-1) * smoothness_std
        world = torch.cat([world, world + noise], dim=0)
        query_points = world.requires_grad_(True)
        sdf, _ = qp_to_sdf(query_points, volume_origin, volume_dim, feat_volume, sdf_decoder, concat_qp=concat_qp_to_sdf)
        normals = compute_grads(sdf, query_points, feat_volume)
        normreg_weights = 1. / (sdf.detach().abs().sum(dim=0) + 1.)
        normal_regularisation_loss = ((normals[0] - normals[1]).norm(dim=-1) * normreg_weights).sum() / normreg_weights.sum()
        # eikonal_loss = torch.cat([eikonal_loss, torch.square(normals.norm(dim=-1) - 1.)])
        # eikonal_loss = torch.square(normals.norm(dim=-1) - 1.)

    normal_supervision_loss = torch.tensor(0., device=z_vals.device)

    ret = {"rgb": rendered_rgb,
           "depth": rendered_depth,
           #    "depth_var": depth_var.detach(),
           "sdf_loss": sdf_loss,
           "fs_loss": fs_loss,
           "sdfs": sdf,
           "surface": surface_loss,
           "weights": weights,
           "normal_regularisation_loss": normal_regularisation_loss,
           "eikonal_loss": eikonal_loss,
           "normal_supervision_loss": normal_supervision_loss,
           }

    return ret


mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x)


def get_pts_in_frustum(points, c2w, K, H, W):
    """
    :param points: [N, 3] points in world coordinates
    :param c2w: [n, 4, 4]
    :param K: [n, 3, 3]
    :param H:
    :param W:
    :return:
    """
    n_views = c2w.shape[0]
    points = points.t().unsqueeze(0).repeat(n_views, 1, 1)  # [n_views, 3, N]
    w2c = torch.inverse(c2w)  # [n_views, 4, 4]
    R, t = w2c[:, :3, :3], w2c[:, :3, 3]
    pts_cam = torch.bmm(R, points) + t[:, :, None]  # [n_views, 3, N]
    pts_uv = torch.bmm(K, pts_cam)  # [n_views, 3, N]
    pz = pts_uv[:, 2, :]  # [n_views, N]
    py = pts_uv[:, 1, :] / pz
    px = pts_uv[:, 0, :] / pz
    mask_batch = (pz > 0) & (px >= 0.) & (px <= W - 1) & (py >= 0.) & (py <= H - 1)
    mask = mask_batch.any(dim=0)  # [N]
    return points[0, :, mask].t()


def from_grid_sample(feat_pts):
    return feat_pts.squeeze(0).squeeze(1).permute(1, 2, 0)

from pdb import set_trace

def qp_to_sdf(pts, volume_origin, volume_dim, feat_volume, sdf_decoder, hyper_embed=None, sdf_act=nn.Identity(),
              concat_qp=False,  rgb_dim=8, use_mask=False, temp_lat_feat=None):
    pts_norm = 2. * (pts - volume_origin[None, None, :]) / volume_dim[None, None, :] - 1.
    mask = (pts_norm.abs() <= 1.)[...,:2].all(dim=-1)
    #set_trace()
    if use_mask:
        pts_norm = pts_norm[mask].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, n_rays, n_samples, 3]
        feat_pts = feat_volume(pts_norm[..., [2, 1, 0]], concat=False)  # [1, feat_dim, 1, n_rays, n_samples]
        inputs = [feat_pts[0][:, :-rgb_dim, ...]] + feat_pts[1:]

        rgb_feat = torch.zeros(*mask.shape, rgb_dim, device=pts_norm.device)
        rgb_feat[mask] = feat_pts[0][0, -rgb_dim:, 0, 0, ...].t()  # [N, 8]

        if concat_qp:
            inputs.append(pts_norm.permute(0, 4, 1, 2, 3))

        features = torch.cat(inputs, dim=1)

        if hyper_embed is not None:
            features = torch.cat((features, hyper_embed), dim=-1)

        raw = sdf_decoder(features.squeeze(0).squeeze(1).squeeze(1).t())
        sdf = torch.zeros_like(mask, dtype=pts_norm.dtype)
        sdf[mask] = sdf_act(raw[..., 0])

        return sdf, rgb_feat, mask
    else:
        pts_norm = pts_norm.unsqueeze(0).unsqueeze(0)  # [1, 1, n_rays, n_samples, 3]
        feat_pts = feat_volume(pts_norm[..., [2, 1, 0]], concat=False)  # [1, feat_dim, 1, n_rays, n_samples]
        inputs = [feat_pts[0][:, :-rgb_dim, ...]] + feat_pts[1:]

        rgb_feat = from_grid_sample(feat_pts[0][:, -rgb_dim:, ...])

        if concat_qp:
            inputs.append(pts_norm.permute(0, 4, 1, 2, 3))

        features = from_grid_sample(torch.cat(inputs, dim=1))

        if hyper_embed is not None:
            if len(features.shape) == 3:
                if features.shape[1] == 1:
                    hyper_embed = hyper_embed.unsqueeze(1)
                else:
                    hyper_embed = hyper_embed.unsqueeze(0)
            features = torch.cat((features, hyper_embed), dim=-1)
        if temp_lat_feat is not None:
            raw = sdf_decoder(features, temp_lat_feat)
        else:
            raw = sdf_decoder(features)
        sdf = sdf_act(raw[..., 0])
        # TODO: why the sdf net is flipping dimension
        return sdf.t(), rgb_feat, mask


def neus_weights(sdf, dists, inv_s, z_vals=None, view_dirs=None, grads=None, cos_val=None, cos_anneal_ratio=0.,
                 batch_size=1024, n_samples=32):
    if cos_val is None:
        cos_val = (view_dirs * grads).sum(-1)
        # cos_val = -F.relu(-cos_val)
        cos_val = -(F.relu(-cos_val * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                    F.relu(-cos_val) * cos_anneal_ratio)

    estimated_next_sdf = sdf + cos_val * dists.reshape(-1, 1) * 0.5
    estimated_prev_sdf = sdf - cos_val * dists.reshape(-1, 1) * 0.5

    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
    #acc_trans = torch.cumprod(torch.cat([torch.ones([sdf.shape[0], 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    acc_trans = torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    weights = alpha * acc_trans
    trans = acc_trans[:, -1]

    if z_vals is not None:
        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0., torch.ones_like(signs), torch.zeros_like(signs))
        # This will only return the first zero-crossing
        inds = torch.argmax(mask, dim=1, keepdim=True)
        z_surf = torch.gather(z_vals, 1, inds)
        return weights, z_surf, alpha, trans

    return weights, alpha, trans


def get_sdf_loss(z_vals, target_d, predicted_sdf, truncation, back_truncation=0.01, inside_masksphere=None):

    if inside_masksphere is not None:
        #sdf_mask *= inside_masksphere
        predicted_sdf = predicted_sdf * inside_masksphere
        z_vals = z_vals * inside_masksphere
        target_d = target_d * inside_masksphere

    depth_mask = target_d > 0.
    front_mask = (z_vals < (target_d - truncation))
    back_mask = (z_vals > (target_d + back_truncation)) & depth_mask
    front_mask = (front_mask | ((target_d < 0.) & (z_vals < 3.5)))
    bound = (target_d - z_vals)
    bound[target_d[:, 0] < 0., :] = 10.  # TODO: maybe use noisy depth for bound?
    sdf_mask = (bound.abs() <= truncation) & depth_mask

    sum_of_samples = front_mask.sum(dim=-1) + sdf_mask.sum(dim=-1) + 1e-8
    # rays_w_depth = torch.count_nonzero(target_d)
    # fs_loss = (torch.max(torch.exp(-5. * predicted_sdf) - 1., predicted_sdf - bound).clamp(min=0.) * front_mask)
    # fs_loss = (fs_loss.sum(dim=-1) / sum_of_samples).sum() / rays_w_depth
    # sdf_loss = ((torch.abs(predicted_sdf - bound) * sdf_mask).sum(dim=-1) / sum_of_samples).sum() / rays_w_depth

    back_fs_samples = back_mask.sum(dim=-1) + 1e-8
    front_fs_loss = (torch.max(torch.exp(-5. * predicted_sdf) - 1., predicted_sdf - bound).clamp(min=0.) * front_mask).sum(dim=-1) / sum_of_samples
    psi_b = 0.6 #*torch.exp(-25. * (z_vals - back_truncation*torch.ones_like(z_vals))**2)
    back_fs_loss = (torch.max(torch.exp(-5. * predicted_sdf) - psi_b, torch.zeros_like(predicted_sdf)).clamp(min=0.) * back_mask).sum(dim=-1) / back_fs_samples
    fs_loss = front_fs_loss + back_fs_loss
    sdf_loss = (torch.abs(predicted_sdf - bound) * sdf_mask).sum(dim=-1) / sum_of_samples

    return fs_loss, sdf_loss


def apply_log_transform(tsdf):
    sgn = torch.sign(tsdf)
    out = torch.log(torch.abs(tsdf) + 1)
    out = sgn * out
    return out


def compute_grads(predicted_sdf, query_points):
    # print("Compute grads")
    # feat_volume.requires_grad_(False)
    sdf_grad, = torch.autograd.grad([predicted_sdf], [query_points], [torch.ones_like(predicted_sdf)], create_graph=True)
    # feat_volume.requires_grad_(True)
    return sdf_grad


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    # device = weights.get_device()
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1], device=device), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / N_importance, 1. - 0.5 / N_importance, steps=N_importance, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)

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
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples