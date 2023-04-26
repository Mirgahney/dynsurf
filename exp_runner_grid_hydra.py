import os
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory

from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, DeformNetwork, AppearanceNetwork, \
    TopoNetwork, SODeformaNet, GeometryNet, RadianceNet, GeometryNetLat, RadianceNetLat
from models.renderer import NeuSRenderer, DeformNeuSRenderer
from models.grid_renderer import DeformGoSRenderer
from models.multi_grid import MultiGrid, GaussianMultiGrid
#from models.encodings import TensorVMEncoding
from models.rend_utils import coordinates, qp_to_sdf
from models.utils import TVLoss
from models.losses import ScaleAndShiftInvariantLoss
from models.weight_scheduler import CosineAnnealingW2
import hydra
from omegaconf import OmegaConf
from pdb import set_trace


class Runner:
    def __init__(self, conf, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')
        self.gpu = torch.cuda.current_device()
        self.dtype = torch.get_default_dtype()

        # Configuration
        self.conf = conf
        self.base_dir = f'{self.conf.general.base_dir}'
        self.base_exp_dir = f'{self.base_dir}{self.conf.general.base_exp_dir}{case}/result'
        #self.conf_path = f'{self.base_dir}/{config_path}/{config_name}'
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(ConfigFactory.from_dict(dict(self.conf['dataset'])))
        self.iter_step = 0

        # Deform
        self.use_deform = self.conf.train.use_deform
        self.use_global_rigid = self.conf.train.use_global_rigid
        self.use_bijective = self.conf.train.use_bijective
        self.use_topo = self.conf.train.use_topo
        if self.use_deform:
            self.deform_dim = self.conf.deformation_network.d_feature
            self.deform_codes = torch.randn(self.dataset.n_loaded_images, self.deform_dim, requires_grad=True).to(
                self.device)
            #self.appearance_dim = self.conf.model.appearance_rendering_network.d_global_feature
            self.appearance_dim = self.conf.deformation_network.d_feature
            self.appearance_codes = torch.randn(self.dataset.n_loaded_images, self.appearance_dim,
                                                requires_grad=True).to(self.device)
            if self.use_global_rigid:
                self.log_qua = nn.parameter.Parameter(torch.zeros((self.dataset.n_loaded_images, 9))).to(
                    self.device)  # [N, 9]
            else:
                self.log_qua = None

        # Training parameters
        self.end_iter = self.conf.train.end_iter
        self.important_begin_iter = self.conf.model.neus_renderer.important_begin_iter
        # Anneal
        self.max_pe_iter = self.conf.train.max_pe_iter

        self.save_freq = self.conf.train.save_freq
        self.report_freq = self.conf.train.report_freq
        self.val_freq = self.conf.train.val_freq
        self.val_mesh_freq = self.conf.train.val_mesh_freq
        self.validate_idx = self.conf.train.validate_idx
        self.batch_size = self.conf.train.batch_size
        self.validate_resolution_level = self.conf.train.validate_resolution_level
        self.learning_rate = self.conf.train.learning_rate
        self.learning_rate_alpha = self.conf.train.learning_rate_alpha
        self.warm_up_end = self.conf.train.warm_up_end
        self.anneal_end = self.conf.train.anneal_end
        self.test_batch_size = self.conf.test.test_batch_size
        self.ndr_color_net = self.conf.train.use_app

        # Weights
        self.igr_weight = self.conf.train.igr_weight
        self.mask_weight = self.conf.train.mask_weight
        self.sdf_weight = self.conf.train.sdf_weight
        self.fs_weight = self.conf.train.fs_weight
        self.surf_sdf = self.conf.train.surf_sdf
        self.tv_weight = self.conf.train.tv_weight
        self.akap_weight = self.conf.train.akap_weight
        self.temp_weight = self.conf.train.temp_weight
        self.latent_weight = self.conf.train.latent_weight
        self.smooth_weight = self.conf.train.smooth_weight
        self.smooth_std = self.conf.train.smooth_std
        self.smooth_std = self.conf.train.smooth_std
        self.smooth_eta = self.conf.train.smooth_eta
        self.smooth_tail = self.conf.train.smooth_tail
        self.normal_l1_weight = self.conf.train.normal_l1_weight
        self.normal_cos_weight = self.conf.train.normal_cos_weight
        self.CosineAnnealingW = CosineAnnealingW2(self.smooth_weight, self.max_pe_iter)

        self.depth_only = self.conf.train.depth_only

        # Scale invariant depth loss
        self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1, lambda_1=1., lambda_2=1e-3)

        self.truncation = self.conf.train.truncation
        self.back_truncation = self.conf.train.back_truncation
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Depth
        self.use_depth = self.conf.dataset.use_depth
        self.use_pred_depth = self.conf.dataset.use_pred_depth
        if self.use_depth or self.use_pred_depth:
            self.geo_weight = self.conf.train.geo_weight
            self.angle_weight = self.conf.train.angle_weight
            self.geo_scale = self.conf.train.geo_scale
            self.rgb_scale = self.conf.train.rgb_scale
            self.regular_scale = self.conf.train.regular_scale

        # Deform
        if self.use_deform:
            if self.use_bijective:
                #self.deform_network = DeformNetwork(**self.conf['model.deform_network']).to(self.device)
                self.deform_network = hydra.utils.instantiate(self.conf.deformation_network).to(self.device)
            else:
                self.deform_network = SODeformaNet(**self.conf['model.deform_network']).to(self.device)
            #self.topo_network = TopoNetwork(**self.conf['model.topo_network']).to(self.device)
            self.topo_network = hydra.utils.instantiate(self.conf.topo_network).to(self.device)
        # self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        #self.sdf_network = GeometryNet(**self.conf['model.sdf_network']).to(self.device)
        self.sdf_network = hydra.utils.instantiate(self.conf.sdf_network).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf.model.variance_network).to(self.device)
        # Deform
        if self.use_deform:
            # self.color_network = AppearanceNetwork(**self.conf['model.appearance_rendering_network']).to(self.device)
            #self.color_network = RadianceNet(**self.conf['model.rendering_network']).to(self.device)
            self.color_network = hydra.utils.instantiate(self.conf.rendering_network).to(self.device)
        else:
            self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)

        # Feature Grid
        self.volume_origin = None
        self.volume_max = None
        self.volume_dim = None
        # xyz_min, xyz_max = torch.tensor([-1.05, -1.05, -1.05]), torch.tensor([1.05, 1.05, 1.05])
        xyz_min = torch.tensor(self.dataset.object_bbox_min, dtype=self.dtype)
        xyz_max = torch.tensor(self.dataset.object_bbox_max, dtype=self.dtype)
        self.init_volume_dims(xyz_min, xyz_max)
        self.feat_dims = self.conf.model.feature_grid.feat_dims
        self.fine_res = self.conf.model.feature_grid.res
        self.grid_type = self.conf.model.feature_grid.type
        self.feature_grid = self.create_feature_grid(self.fine_res, self.feat_dims)
        self.rgb_dim = self.conf.model.feature_grid.rgb_dim
        self.pg_scale = self.conf.model.feature_grid.pg_scale
        self.feat_scale = self.conf.model.feature_grid.feat_scale

        # Deform
        if self.use_deform:
            self.renderer = DeformGoSRenderer(self.report_freq,
                                              self.deform_network,
                                              self.topo_network,
                                              self.sdf_network,
                                              self.deviation_network,
                                              self.color_network,
                                              self.feature_grid,
                                              self.volume_origin,
                                              self.volume_dim,
                                              self.rgb_dim,
                                              ndr_color_net=self.ndr_color_net,
                                              **self.conf.model.neus_renderer)
        else:
            self.renderer = NeuSRenderer(self.sdf_network,
                                         self.deviation_network,
                                         self.color_network,
                                         **self.conf.model.neus_renderer)

        # Load Optimizer
        params_to_train = []
        if self.use_deform:
            params_to_train += [
                {'name': 'deform_network', 'params': self.deform_network.parameters(), 'lr': self.learning_rate}]
            params_to_train += [
                {'name': 'topo_network', 'params': self.topo_network.parameters(), 'lr': self.learning_rate}]
            params_to_train += [{'name': 'deform_codes', 'params': self.deform_codes, 'lr': self.learning_rate}]
            params_to_train += [{'name': 'appearance_codes', 'params': self.appearance_codes, 'lr': self.learning_rate}]
            if self.use_global_rigid:
                params_to_train += [{'name': 'log_qua', 'params': self.log_qua, 'lr': 7e-4}]
        params_to_train += [
            {'name': 'feature_grid', 'params': self.feature_grid.parameters(), 'lr': self.learning_rate}]
        params_to_train += [{'name': 'sdf_network', 'params': self.sdf_network.parameters(), 'lr': self.learning_rate}]
        params_to_train += [
            {'name': 'deviation_network', 'params': self.deviation_network.parameters(), 'lr': self.learning_rate}]
        params_to_train += [
            {'name': 'color_network', 'params': self.color_network.parameters(), 'lr': self.learning_rate}]

        # Camera
        if self.dataset.camera_trainable:
            params_to_train += [
                {'name': 'intrinsics_paras', 'params': self.dataset.intrinsics_paras, 'lr': self.learning_rate}]
            params_to_train += [{'name': 'poses_paras', 'params': self.dataset.poses_paras, 'lr': self.learning_rate}]
            # Depth
            if self.use_depth:
                params_to_train += [{'name': 'depth_intrinsics_paras', 'params': self.dataset.depth_intrinsics_paras,
                                     'lr': self.learning_rate}]

        self.optimizer = torch.optim.Adam(params_to_train)
        self.tv_loss_fn = TVLoss(1.0, [1])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            if self.mode == 'validate_pretrained':
                latest_model_name = 'pretrained.pth'
            else:
                model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
                model_list = []
                for model_name in model_list_raw:
                    if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                        model_list.append(model_name)
                model_list.sort()
                latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs
        if self.mode[:5] == 'train':
            self.file_backup()
            # self.geom_init(2.1/5)
            if not self.is_continue:
                # self.geom_init(2.1/2.3)
                self.geom_init(2.1 / 4.)
                #pass

    def update_and_create_optimizer(self):
        optim_param_groups = self.optimizer.param_groups
        # Load Optimizer
        params_to_train = []
        i = 0
        if self.use_deform:
            assert 'deform_network' == optim_param_groups[i]['name']
            params_to_train += [{'name': 'deform_network', 'params': self.deform_network.parameters(),
                                 'lr': optim_param_groups[i]['lr']}]
            i += 1
            params_to_train += [{'name': 'topo_network', 'params': self.topo_network.parameters(),
                                 'lr': optim_param_groups[i]['lr']}]
            i += 1
            params_to_train += [
                {'name': 'deform_codes', 'params': self.deform_codes, 'lr': optim_param_groups[i]['lr']}]
            i += 1
            params_to_train += [{'name': 'appearance_codes', 'params': self.appearance_codes,
                                 'lr': optim_param_groups[i]['lr']}]
            i += 1
            if self.use_global_rigid:
                assert 'log_qua' == optim_param_groups[i]['name']
                params_to_train += [{'name': 'log_qua', 'params': self.log_qua, 'lr': optim_param_groups[i]['lr']}]
                i += 1
        params_to_train += [{'name': 'feature_grid', 'params': self.feature_grid.parameters(),
                             'lr': optim_param_groups[i]['lr']}]
        i += 1
        params_to_train += [{'name': 'sdf_network', 'params': self.sdf_network.parameters(),
                             'lr': optim_param_groups[i]['lr']}]
        i += 1
        params_to_train += [{'name': 'deviation_network', 'params': self.deviation_network.parameters(),
                             'lr': optim_param_groups[i]['lr']}]
        i += 1
        params_to_train += [{'name': 'color_network', 'params': self.color_network.parameters(),
                             'lr': optim_param_groups[i]['lr']}]
        i += 1

        # Camera
        if self.dataset.camera_trainable:
            assert 'intrinsics_paras' == optim_param_groups[i]['name']
            params_to_train += [{'name': 'intrinsics_paras', 'params': self.dataset.intrinsics_paras,
                                 'lr': optim_param_groups[i]['lr']}]
            i += 1
            params_to_train += [{'name': 'poses_paras', 'params': self.dataset.poses_paras,
                                 'lr': optim_param_groups[i]['lr']}]
            i += 1
            # Depth
            if self.use_depth:
                assert 'depth_intrinsics_paras' == optim_param_groups[i]['name']
                params_to_train += [{'name': 'depth_intrinsics_paras', 'params': self.dataset.depth_intrinsics_paras,
                                     'lr': optim_param_groups[i]['lr']}]
                i += 1

        self.optimizer = torch.optim.Adam(params_to_train)

    def create_feature_grid(self, fine_res, feat_dims):
        voxel_dims = []
        for i in range(len(feat_dims)):
            res = fine_res // 2 ** i
            voxel_dims.append(torch.tensor([res + 1] * 3, device=self.device))
        self.world_size = voxel_dims
        voxel_dims = torch.stack(voxel_dims, 0)
        # init_feat = torch.rand(1, torch.tensor(feat_dims).sum(), device=self.device) * 0.02 - 0.01
        init_feat = torch.zeros(1, torch.tensor(feat_dims).sum(), device=self.device)
        if self.grid_type == 'normal':
            return MultiGrid(voxel_dims, init_feat, feat_dims)
        elif self.grid_type == 'Gaussian':
            return GaussianMultiGrid(voxel_dims, init_feat, feat_dims)
        #elif self.grid_type == 'TensoRF_VM':
        #    return TensorVMEncoding(128, 11, smoothstep=True)

    def _set_grid_resolution(self, feat_scale):
        # Determine grid resolution
        voxel_dims = []
        for i in range(len(self.world_size)):
            res = self.world_size[i][0]
            res_scale = feat_scale[i]
            voxel_dims.append(torch.tensor([int(res * res_scale)] * 3, device=self.device))
        self.world_size = voxel_dims

    def upscale_feature_grid(self, feat_scale):
        print('scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(feat_scale)
        print('scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)
        self.feature_grid.upscale(self.world_size)

    def tv_loss(self, volums):
        total = 0.0
        for vol in volums:
            total = total + self.tv_loss_fn(vol)
        return total

    # depth loss based on ScaleAndShiftInvariantLoss
    def get_depth_loss(self, depth_pred, depth_gt, mask):
        # TODO remove hard-coded scaling for depth
        return self.depth_loss(depth_pred.reshape(1, 32, 32), (depth_gt).reshape(1, 32, 32), #(depth_gt * 20 + 0.5).reshape(1, 32, 32),
                                   mask.reshape(1, 32, 32))

    def geom_init(self, radius=1.5, chunk=500000):
        optimizer = torch.optim.Adam([
            {"params": self.sdf_network.parameters(), "lr": 1e-3},
            {"params": self.topo_network.parameters(), "lr": 1e-3},
            {"params": self.feature_grid.parameters(), "lr": 5e-2},
            {"params": self.deform_codes, "lr": 1e-3},
        ])
        center = self.volume_origin + self.volume_dim / 2.
        # TODO: radius cannot be too large! FOV must cover the sphere, and there must be enough background space
        # radius = 0.8 * self.volume_dim.min() / 2
        offset = torch.tensor([0., 0., 0.], device=self.device)
        center += offset
        iters = 1500
        print_i = iters // 5
        alpha_ratio = 0

        for j in range(iters):
            optimizer.zero_grad()
            coords = coordinates([self.feature_grid.volumes[0].shape[2],
                                  self.feature_grid.volumes[0].shape[3],
                                  self.feature_grid.volumes[0].shape[4]], self.device).float().t()  # [N, 3]
            voxel_size_xyz = self.volume_dim / (torch.tensor(self.feature_grid.volumes[0].shape[2:]) - 1)  # [3,]

            qp = (coords * voxel_size_xyz[None, :] + self.volume_origin[None, :])
            rand_idx = torch.randint(0, self.dataset.n_loaded_images, (1,))
            deform_code = self.deform_codes[rand_idx]
            log_qua = self.log_qua[rand_idx] if self.use_global_rigid else None
            target_sdf = (center - qp).norm(dim=-1) - radius
            n_voxels = qp.shape[0]
            for i in range(0, n_voxels, chunk):
                start = i
                end = i + chunk if (i + chunk) <= n_voxels else n_voxels
                qp_chunk = qp[start:end, :]
                target_sdf_chunk = target_sdf[start:end]
                ambient_coord = self.topo_network(deform_code, qp_chunk, alpha_ratio)

                sdf, *_ = qp_to_sdf(qp_chunk.unsqueeze(1), self.volume_origin, self.volume_dim, self.feature_grid,
                                    self.sdf_network, hyper_embed=ambient_coord, rgb_dim=self.rgb_dim)
                # sdf = sdf.squeeze(-1)
                sdf = sdf.squeeze(0)

                loss = torch.nn.functional.mse_loss(sdf, target_sdf_chunk)
                # loss = torch.nn.functional.l1_loss(sdf, target_sdf_chunk)
                loss.backward(retain_graph=True)
                optimizer.step()
            if j % print_i == 0:
                print("SDF loss: {}".format(loss.item()))

    def init_volume_dims(self, xyz_min, xyz_max):
        self.volume_origin = xyz_min
        self.volume_max = xyz_max
        self.volume_dim = xyz_max - xyz_min

    def train(self):
        tb_dir = os.path.join(self.base_exp_dir, 'logs')
        self.writer = SummaryWriter(log_dir=tb_dir)
        self.profile = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_dir),
            record_shapes=True, profile_memory=True, with_stack=True)
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        # self.profile.start()
        for iter_i in tqdm(range(res_step)):
            # Deform
            if self.use_deform:
                image_idx = image_perm[self.iter_step % len(image_perm)]
                # Deform
                deform_code = self.deform_codes[image_idx][None, ...]
                log_qua = self.log_qua[image_idx][None, ...] if self.use_global_rigid else None
                if iter_i == 0:
                    print('The files will be saved in:', self.base_exp_dir)
                    print('Used GPU:', self.gpu)
                    #self.validate_observation_mesh(self.validate_idx)
                    # self.validate_canonical_mesh()
                    #self.validate_all_mesh()

                if iter_i in self.pg_scale:
                    self.upscale_feature_grid(self.feat_scale)
                    self.update_and_create_optimizer()

                # Deptdeviation_networkh
                if self.use_pred_depth:
                    data = self.dataset.gen_random_rays_at_depth(image_idx, self.batch_size)
                    rays_o, rays_d, rays_s, rays_l, true_rgb, mask, normal = \
                        data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10], data[:, 10: 13], data[:, 13: 14], \
                        data[:, 14: 17]
                elif self.use_depth:
                    data = self.dataset.gen_random_rays_at_depth(image_idx, self.batch_size)
                    rays_o, rays_d, rays_s, rays_l, true_rgb, mask = \
                        data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10], data[:, 10: 13], data[:, 13: 14]
                    normal = None
                else:
                    data = self.dataset.gen_random_rays_at(image_idx, self.batch_size)
                    rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
                    rays_l = None

                # Deform
                appearance_code = self.appearance_codes[image_idx][None, ...]
                # Anneal
                alpha_ratio = max(min(self.iter_step / self.max_pe_iter, 1.), 0.)

                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

                if self.mask_weight > 0.0:
                    mask = (mask > 0.5).to(self.dtype)
                else:
                    mask = torch.ones_like(mask)
                mask_sum = mask.sum() + 1e-5
                if self.sdf_weight != 0.0:
                    render_out = self.renderer.render(deform_code, appearance_code, rays_o, rays_d, near, far,
                                                      cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                      alpha_ratio=alpha_ratio, iter_step=self.iter_step, rays_l=rays_l,
                                                      truncation=self.truncation, back_truncation=self.back_truncation,
                                                      log_qua=log_qua, gt_normals=normal)
                else:
                    render_out = self.renderer.render(deform_code, appearance_code, rays_o, rays_d, near, far,
                                                      cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                      alpha_ratio=alpha_ratio, iter_step=self.iter_step)
                # Depth
                if self.use_depth or self.use_pred_depth:
                    sdf_loss_surf, angle_loss, valid_depth_region, normal_smoothness_loss = \
                        self.renderer.errorondepth(deform_code, rays_o, rays_d, rays_s, mask,
                                                   alpha_ratio, iter_step=self.iter_step, log_qua=log_qua,
                                                   smooth_std=self.smooth_std)
                    foreground_mask = mask[:, 0] > 0.5
                    inside_sphere = render_out['inside_sphere']
                    if self.sdf_weight != 0.0:
                        sdf_loss = 10 * render_out['sdf_loss'][foreground_mask].mean()
                        if self.surf_sdf != 0.0:
                            sdf_loss = sdf_loss + self.surf_sdf * sdf_loss_surf
                    else:
                        sdf_loss = sdf_loss_surf
                    fs_loss = render_out['fs_loss'][foreground_mask].mean()

                color_fine = render_out['color_fine']
                s_val = render_out['s_val']
                cdf_fine = render_out['cdf_fine']
                gradient_o_error = render_out['gradient_o_error']
                weight_max = render_out['weight_max']
                weight_sum = render_out['weight_sum']
                depth_map = render_out['depth_map']
                akap_loss = render_out['akap_loss']
                elastic_loss = render_out['elastic_loss']
                normal_l1 = render_out['normal_l1']
                normal_cos = render_out['normal_cos']

                # Loss
                color_error = (color_fine - true_rgb) * mask
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

                eikonal_loss = gradient_o_error

                mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-5, 1.0 - 1e-5), mask)
                #mask_loss = 0.0 #F.binary_cross_entropy(weight_sum.clip(1e-5, 1.0 - 1e-3), mask)
                # Depth
                if self.use_depth or self.use_pred_depth:
                    if self.use_pred_depth:
                        #depth_loss = self.get_depth_loss(depth_map, rays_l, valid_depth_region)
                        #depth_loss, scale_loss = self.get_depth_loss(depth_map, rays_l, mask)  #* 0.2  # hard value to make the loss ~ 1.1
                        #depth_loss, scale_loss = self.get_depth_loss(depth_map, rays_l, torch.ones_like(rays_l))  #* 0.2  # hard value to make the loss ~ 1.1
                        # until we fix the valid_depth_mask

                        depth_minus = (depth_map - rays_l) * valid_depth_region
                        depth_loss = F.l1_loss(depth_minus, torch.zeros_like(depth_minus), reduction='sum') \
                                    / (valid_depth_region.sum() + 1e-5)
                        scale_loss = 0.0
                    else:
                        depth_minus = (depth_map - rays_l) * valid_depth_region
                        depth_loss = F.l1_loss(depth_minus, torch.zeros_like(depth_minus), reduction='sum') \
                                     / (valid_depth_region.sum() + 1e-5)
                        scale_loss = 0.0
                        #depth_loss, scale_loss = self.get_depth_loss(depth_map, rays_l, mask)  # * 0.2  # hard value to make the loss ~ 1.1
                    if self.iter_step < self.important_begin_iter:
                        rgb_scale = self.rgb_scale[0]  # 0.1
                        geo_scale = self.geo_scale[0]  # 10.0
                        regular_scale = self.regular_scale[0]  # 10.0
                        geo_loss = sdf_loss
                    elif self.iter_step < self.max_pe_iter:
                        rgb_scale = self.rgb_scale[1]  # 1.0
                        geo_scale = self.geo_scale[1]  # 1.0
                        regular_scale = self.regular_scale[1]  # 10.0
                        geo_loss = 0.5 * (depth_loss + sdf_loss)
                    else:
                        rgb_scale = self.rgb_scale[2]  # 1.0
                        geo_scale = self.geo_scale[2]  # 0.1  # big jump for our case maybe?
                        regular_scale = self.regular_scale[2]  # 1.0
                        geo_loss = 0.5 * (depth_loss + sdf_loss)
                else:
                    if self.iter_step < self.max_pe_iter:
                        regular_scale = 10.0
                    else:
                        regular_scale = 1.0

                if self.depth_only:
                    geo_loss = depth_loss
                    #geo_loss = 0.0
                    #scale_loss = 0.0

                tv_loss = self.tv_loss(self.feature_grid.volumes)

                if self.use_depth or self.use_pred_depth:
                    loss = color_fine_loss * rgb_scale + \
                           (geo_loss * self.geo_weight + angle_loss * self.angle_weight + 1.0 * scale_loss) * geo_scale + \
                           (eikonal_loss * self.igr_weight + mask_loss * self.mask_weight + akap_loss * self.akap_weight
                            + normal_smoothness_loss * self.smooth_weight #+ normal_l1 * self.normal_l1_weight
                            #+ normal_cos * self.normal_cos_weight
                            ) * regular_scale + fs_loss * self.fs_weight + tv_loss * self.tv_weight + \
                           normal_l1 * self.normal_l1_weight + normal_cos * self.normal_cos_weight
                    # + akap_loss * self.akap_weight
                else:
                    loss = color_fine_loss + \
                           (eikonal_loss * self.igr_weight + mask_loss * self.mask_weight) * regular_scale
                    # loss = (eikonal_loss * self.igr_weight + mask_loss * self.mask_weight) * regular_scale

                # temporal regularisation
                N = self.deform_codes.shape[0]
                ids = torch.arange(N).long().to(self.device)
                prev_delta = torch.randint(1, 3, (N,)).to(self.device)
                prev_ids = torch.clamp(ids - prev_delta, min=0).long()
                next_delta = torch.randint(1, 3, (N,)).to(self.device)
                next_ids = torch.clamp(ids + next_delta, max=N - 1).long()
                deform_temporal_loss, deform_code_loss, app_temporal_loss, app_code_loss = \
                    self.get_latent_loss(prev_ids, next_ids)
                temporal_loss = deform_temporal_loss + app_temporal_loss
                code_loss = deform_code_loss + app_code_loss

                loss = loss + self.temp_weight * temporal_loss + self.latent_weight * code_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.iter_step += 1

                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
                del color_fine_loss
                # Depth
                if self.use_depth:
                    self.writer.add_scalar('Loss/sdf_loss', sdf_loss, self.iter_step)
                    self.writer.add_scalar('Loss/fs_loss', fs_loss, self.iter_step)
                    self.writer.add_scalar('Loss/depth_loss', depth_loss, self.iter_step)
                    self.writer.add_scalar('Loss/angle_loss', angle_loss, self.iter_step)
                    self.writer.add_scalar('Loss/normal_l1', normal_l1, self.iter_step)
                    self.writer.add_scalar('Loss/normal_cos', normal_cos, self.iter_step)
                    self.writer.add_scalar('Loss/scale_loss', scale_loss, self.iter_step)
                    del sdf_loss
                    del depth_loss
                    del angle_loss
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
                self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
                self.writer.add_scalar('Loss/tv_loss', tv_loss, self.iter_step)
                self.writer.add_scalar('Loss/akap_loss', akap_loss, self.iter_step)
                self.writer.add_scalar('Loss/elastic_loss', elastic_loss, self.iter_step)
                self.writer.add_scalar('Loss/smoothness_loss', normal_smoothness_loss, self.iter_step)
                del eikonal_loss
                del mask_loss

                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

                if self.iter_step % self.report_freq == 0:
                    print('The files have been saved in:', self.base_exp_dir)
                    print('Used GPU:', self.gpu)
                    print('iter:{:8>d} loss={} idx={} alpha_ratio={} lr={}'.format(self.iter_step, loss, image_idx,
                                                                                   alpha_ratio,
                                                                                   self.optimizer.param_groups[0][
                                                                                       'lr']))

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()

                if self.iter_step % self.val_freq == 0:
                    self.validate_image(self.validate_idx)
                    # Depth
                    if self.use_depth or self.use_pred_depth:
                        self.validate_image_with_depth(self.validate_idx)

                if self.iter_step % self.val_mesh_freq == 0:
                    self.validate_observation_mesh(self.validate_idx)

                self.update_learning_rate()

                if self.iter_step % len(image_perm) == 0:
                    image_perm = self.get_image_perm()

            else:
                if self.iter_step == 0:
                    self.validate_mesh()
                data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

                rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
                near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

                if self.mask_weight > 0.0:
                    mask = (mask > 0.5).to(self.dtype)
                else:
                    mask = torch.ones_like(mask)

                mask_sum = mask.sum() + 1e-5
                render_out = self.renderer.render(rays_o, rays_d, near, far,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio())

                color_fine = render_out['color_fine']
                s_val = render_out['s_val']
                cdf_fine = render_out['cdf_fine']
                gradient_error = render_out['gradient_error']
                weight_max = render_out['weight_max']
                weight_sum = render_out['weight_sum']
                sdf_loss = render_out['sdf_loss']
                fs_loss = render_out['fs_loss']

                # Loss
                color_error = (color_fine - true_rgb) * mask
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())

                eikonal_loss = gradient_error

                mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)

                loss = color_fine_loss + \
                       eikonal_loss * self.igr_weight + \
                       mask_loss * self.mask_weight  # + sdf_loss * self.sdf_weight + fs_loss * self.fs_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.iter_step += 1

                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
                del color_fine_loss
                del eikonal_loss
                if self.mask_weight > 0.0:
                    self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
                    del mask_loss
                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

                if self.iter_step % self.report_freq == 0:
                    print('The file have been saved in:', self.base_exp_dir)
                    print('Used GPU:', self.gpu)
                    print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss,
                                                               self.optimizer.param_groups[0]['lr']))

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()

                if self.iter_step % self.val_freq == 0:
                    self.validate_image()

                if self.iter_step % self.val_mesh_freq == 0:
                    self.validate_mesh()

                self.update_learning_rate()

                if self.iter_step % len(image_perm) == 0:
                    image_perm = self.get_image_perm()

            # self.profile.step()
        # self.profile.stop()

    def get_latent_loss(self, prev_ids, next_ids):
        curr_codes = self.deform_codes.data
        prev_codes = self.deform_codes.data[prev_ids, :]
        next_codes = self.deform_codes.data[next_ids, :]
        deform_temporal_loss = torch.abs(next_codes - curr_codes).mean() + torch.abs(curr_codes - prev_codes).mean()
        deform_code_loss = curr_codes.norm(dim=1).mean()

        #curr_codes = self.model.appearance_code.data
        #prev_codes = self.model.appearance_code.data[prev_ids, :]
        #next_codes = self.model.appearance_code.data[next_ids, :]
        app_temporal_loss = 0.0 #torch.abs(next_codes - curr_codes).mean() + torch.abs(curr_codes - prev_codes).mean()
        app_code_loss = 0.0 #curr_codes.norm(dim=1).mean()

        return deform_temporal_loss, deform_code_loss, app_temporal_loss, app_code_loss

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_loaded_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self, scale_factor=1):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        learning_factor *= scale_factor

        current_learning_rate = self.learning_rate * learning_factor
        for g in self.optimizer.param_groups:
            if g['name'] in ['intrinsics_paras', 'poses_paras', 'depth_intrinsics_paras']:
                g['lr'] = current_learning_rate * 1e-1
            elif self.iter_step >= self.max_pe_iter and g['name'] == 'deviation_network':
                g['lr'] = current_learning_rate * 1.5
            else:
                g['lr'] = current_learning_rate

    def file_backup(self):
        dir_lis = self.conf.general.recording
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            dir_name_join = os.path.join(self.base_dir, dir_name)
            files = os.listdir(dir_name_join)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name_join, f_name), os.path.join(cur_dir, f_name))
        OmegaConf.save(self.conf, os.path.join(self.base_exp_dir, 'recording', 'config_full.yaml'))
        logging.info('File Saved')

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        for i in range(len(self.pg_scale)):
            self.upscale_feature_grid(self.feat_scale)
        self.feature_grid.load_state_dict(checkpoint['feature_grid'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        # Deform
        if self.use_deform:
            self.deform_network.load_state_dict(checkpoint['deform_network'])
            self.topo_network.load_state_dict(checkpoint['topo_network'])
            self.deform_codes = torch.from_numpy(checkpoint['deform_codes']).to(self.device).requires_grad_()
            self.appearance_codes = torch.from_numpy(checkpoint['appearance_codes']).to(self.device).requires_grad_()
            logging.info('Use_deform True')
            if self.use_global_rigid:
                self.log_qua = torch.from_numpy(checkpoint['log_qua']).to(self.device).requires_grad_()
                logging.info('Use_global_rigid True')

        self.dataset.intrinsics_paras = torch.from_numpy(checkpoint['intrinsics_paras']).to(self.device)
        self.dataset.poses_paras = torch.from_numpy(checkpoint['poses_paras']).to(self.device)
        # Depth
        if self.use_depth:
            self.dataset.depth_intrinsics_paras = torch.from_numpy(checkpoint['depth_intrinsics_paras']).to(self.device)
        # Camera
        if self.dataset.camera_trainable:
            self.dataset.intrinsics_paras.requires_grad_()
            self.dataset.poses_paras.requires_grad_()
            # Depth
            if self.use_depth:
                self.dataset.depth_intrinsics_paras.requires_grad_()
        else:
            self.dataset.static_paras_to_mat()
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        # Depth
        if self.use_depth:
            depth_intrinsics_paras = self.dataset.depth_intrinsics_paras.data.cpu().numpy()
        else:
            depth_intrinsics_paras = self.dataset.intrinsics_paras.data.cpu().numpy()
        # Deform
        if self.use_deform:
            checkpoint = {
                'deform_network': self.deform_network.state_dict(),
                'topo_network': self.topo_network.state_dict(),
                'feature_grid': self.feature_grid.state_dict(),
                'sdf_network_fine': self.sdf_network.state_dict(),
                'variance_network_fine': self.deviation_network.state_dict(),
                'color_network_fine': self.color_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_step': self.iter_step,
                'deform_codes': self.deform_codes.data.cpu().numpy(),
                'log_qua': self.log_qua.data.cpu().numpy() if self.use_global_rigid else None,
                'appearance_codes': self.appearance_codes.data.cpu().numpy(),
                'intrinsics_paras': self.dataset.intrinsics_paras.data.cpu().numpy(),
                'poses_paras': self.dataset.poses_paras.data.cpu().numpy(),
                'depth_intrinsics_paras': depth_intrinsics_paras,
            }
        else:
            checkpoint = {
                'feature_grid': self.feature_grid.state_dict(),
                'sdf_network_fine': self.sdf_network.state_dict(),
                'variance_network_fine': self.deviation_network.state_dict(),
                'color_network_fine': self.color_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_step': self.iter_step,
                'intrinsics_paras': self.dataset.intrinsics_paras.data.cpu().numpy(),
                'poses_paras': self.dataset.poses_paras.data.cpu().numpy(),
                'depth_intrinsics_paras': depth_intrinsics_paras,
            }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>7d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1, mode='train', normal_filename='normals', rgb_filename='rgbs',
                       depth_filename='depths'):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_loaded_images)
        # Deform
        if self.use_deform:
            deform_code = self.deform_codes[idx][None, ...]
            log_qua = self.log_qua[idx][None, ...] if self.use_global_rigid else None
            appearance_code = self.appearance_codes[idx][None, ...]
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        if mode == 'train':
            batch_size = self.batch_size
        else:
            batch_size = self.test_batch_size

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(batch_size)
        rays_d = rays_d.reshape(-1, 3).split(batch_size)

        out_rgb_fine = []
        out_rgb_fine_ours = []
        out_normal_fine = []
        out_depth_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            if self.use_deform:
                render_out = self.renderer.render(deform_code,
                                                  appearance_code,
                                                  rays_o_batch,
                                                  rays_d_batch,
                                                  near,
                                                  far,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                  alpha_ratio=max(min(self.iter_step / self.max_pe_iter, 1.), 0.),
                                                  iter_step=self.iter_step,
                                                  log_qua=log_qua)
                render_out['gradients'] = render_out['gradients_o']
            else:
                render_out = self.renderer.render(rays_o_batch,
                                                  rays_d_batch,
                                                  near,
                                                  far,
                                                  cos_anneal_ratio=self.get_cos_anneal_ratio())

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                out_rgb_fine_ours.append(render_out['color_ours'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                if self.iter_step >= self.important_begin_iter:
                    n_samples = self.renderer.n_samples + self.renderer.n_importance
                else:
                    n_samples = self.renderer.n_samples
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            #del render_out['depth_map']  # Annotate it if you want to visualize estimated depth map!
            if feasible('depth_map'):
                out_depth_fine.append(render_out['depth_map'].detach().cpu().numpy())
            del render_out

        img_fine = None
        img_fine_ours = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
            img_fine_ours = (np.concatenate(out_rgb_fine_ours, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            # Camera
            if self.dataset.camera_trainable:
                _, pose = self.dataset.dynamic_paras_to_mat(idx)
            else:
                pose = self.dataset.poses_all[idx]
            rot = np.linalg.inv(pose[:3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        depth_img = None
        if len(out_depth_fine) > 0:
            depth_img = np.concatenate(out_depth_fine, axis=0)
            depth_img = depth_img.reshape([H, W, 1, -1])
            depth_img = 255. - np.clip(depth_img / depth_img.max(), a_max=1, a_min=0) * 255.
            depth_img = np.uint8(depth_img)
        os.makedirs(os.path.join(self.base_exp_dir, rgb_filename), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, normal_filename), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, depth_filename), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        rgb_filename,
                                        '{:0>8d}_{:0>3d}.png'.format(self.iter_step, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        rgb_filename,
                                        'ours{:0>8d}_{:0>3d}.png'.format(self.iter_step, idx)),
                           np.concatenate([img_fine_ours[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        normal_filename,
                                        '{:0>8d}_{:0>3d}.png'.format(self.iter_step, idx)),
                           normal_img[..., i])

            if len(out_depth_fine) > 0:
                if self.use_depth:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            depth_filename,
                                            '{:0>8d}_{:0>3d}.png'.format(self.iter_step, idx)),
                               np.concatenate([cv.applyColorMap(depth_img[..., i], cv.COLORMAP_JET),
                                               self.dataset.depth_at(idx, resolution_level=resolution_level)]))
                else:
                    cv.imwrite(os.path.join(self.base_exp_dir, depth_filename,
                                            '{:0>8d}_{:0>3d}.png'.format(self.iter_step, idx)),
                               cv.applyColorMap(depth_img[..., i], cv.COLORMAP_JET))

    def validate_image_with_depth(self, idx=-1, resolution_level=-1, mode='train'):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_loaded_images)

        # Deform
        if self.use_deform:
            deform_code = self.deform_codes[idx][None, ...]
            log_qua = self.log_qua[idx][None, ...] if self.use_global_rigid else None
            appearance_code = self.appearance_codes[idx][None, ...]
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        if mode == 'train':
            batch_size = self.batch_size
        else:
            batch_size = self.test_batch_size

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, rays_s, mask = self.dataset.gen_rays_at_depth(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(batch_size)
        rays_d = rays_d.reshape(-1, 3).split(batch_size)
        rays_s = rays_s.reshape(-1, 3).split(batch_size)
        mask = (mask > 0.5).to(self.dtype).detach().cpu().numpy()[..., None]

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch, rays_s_batch in zip(rays_o, rays_d, rays_s):
            color_batch, gradients_batch = self.renderer.renderondepth(deform_code,
                                                                       appearance_code,
                                                                       rays_o_batch,
                                                                       rays_d_batch,
                                                                       rays_s_batch,
                                                                       alpha_ratio=max(
                                                                           min(self.iter_step / self.max_pe_iter, 1.),
                                                                           0.),
                                                                       log_qua=log_qua)

            out_rgb_fine.append(color_batch.detach().cpu().numpy())
            out_normal_fine.append(gradients_batch.detach().cpu().numpy())
            del color_batch, gradients_batch

        img_fine = None
        if len(out_rgb_fine) > 0:
            # set_trace()
            if self.ndr_color_net:
                img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
            else:
                img_fine = (np.concatenate(out_rgb_fine, axis=1).reshape([H, W, 3, -1]) * 256).clip(0, 255)
            img_fine = img_fine * mask

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            # w/ pose -> w/o pose. similar: world -> camera
            # Camera
            if self.dataset.camera_trainable:
                _, pose = self.dataset.dynamic_paras_to_mat(idx)
            else:
                pose = self.dataset.poses_all[idx]
            rot = np.linalg.inv(pose[:3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
            normal_img = normal_img * mask

        os.makedirs(os.path.join(self.base_exp_dir, 'rgbsondepth'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normalsondepth'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'rgbsondepth',
                                        '{:0>8d}_depth_{:0>3d}.png'.format(self.iter_step, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normalsondepth',
                                        '{:0>8d}_depth_{:0>3d}.png'.format(self.iter_step, idx)),
                           normal_img[..., i])

    def validate_all_image(self, resolution_level=-1):
        for image_idx in range(self.dataset.n_loaded_images):
        #for image_idx in range(100, 2):
            self.validate_image(image_idx, resolution_level, 'test', 'validations_normals', 'validations_rgbs',
                                'validations_depths')
            print('Used GPU:', self.gpu)

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=self.dtype)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=self.dtype)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    # Deform
    def validate_canonical_mesh(self, world_space=False, resolution=64, threshold=0.0, filename='meshes_canonical'):
        print('extract canonical mesh')
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=self.dtype)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=self.dtype)

        vertices, triangles = \
            self.renderer.extract_canonical_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold,
                                                     alpha_ratio=max(min(self.iter_step / self.max_pe_iter, 1.), 0.))
        os.makedirs(os.path.join(self.base_exp_dir, filename), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, filename, '{:0>8d}_canonical.ply'.format(self.iter_step)))

        logging.info('End')

    # Deform
    def validate_observation_mesh(self, idx=-1, world_space=False, resolution=64, threshold=0.0, filename='meshes'):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_loaded_images)
        # Deform
        deform_code = self.deform_codes[idx][None, ...]
        log_qua = self.log_qua[idx][None, ...] if self.use_global_rigid else None

        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=self.dtype)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=self.dtype)
        # bound_min, bound_max = torch.tensor([-1.05, -1.05, -1.05]), torch.tensor([1.05, 1.05, 1.05])

        vertices, triangles = \
            self.renderer.extract_observation_geometry(deform_code, bound_min, bound_max, resolution=resolution,
                                                       threshold=threshold,
                                                       alpha_ratio=max(min(self.iter_step / self.max_pe_iter, 1.), 0.),
                                                       log_qua=log_qua)
        os.makedirs(os.path.join(self.base_exp_dir, filename), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, filename, '{:0>8d}_{:0>3d}.ply'.format(self.iter_step, idx)))

        logging.info('End')

    # Deform
    def validate_all_mesh(self, world_space=False, resolution=64, threshold=0.0):
        for image_idx in range(self.dataset.n_loaded_images):
        #for image_idx in range(100):
            self.validate_observation_mesh(image_idx, world_space, resolution, threshold, 'validations_meshes')
            print('Used GPU:', self.gpu)


config_paths = ["./confs/ddeform", "./confs/killfusion"]
config_names = ['seq004_grid_local.yaml', 'frog_grid_local.yaml', 'seq004_grid.yaml', 'seq004_grid_faster.yaml',
                'seq004_grid_ffaster.yaml']
#'frog_grid_jade.yaml'  #'frog_grid_priyor_fast.yaml'  #'frog_grid_priyor.yaml'#'seq011_grid.yaml'  #'seq004_dog_grid.yaml'  #
@hydra.main(config_path=config_paths[0], config_name=config_names[-3])
def main(cfg):
    #cfg = eval_str_num(cfg)

    print('Welcome to DySurf')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    torch.set_default_dtype(torch.float32)
    torch.cuda.set_device(0)
    mpek = cfg.train.max_pe_iter//1000
    model_name = f'{cfg.case}_mpe{mpek}k_smooth{cfg.train.smooth_weight}_std{cfg.train.smooth_std}_lattemp{cfg.train.temp_weight}' \
                 f'fs{cfg.train.fs_weight}_normal_l1{cfg.train.normal_l1_weight}_cos{cfg.train.normal_cos_weight}'

    runner = Runner(cfg, cfg.mode, model_name, cfg.is_continue)

    if cfg.mode == 'train':
        runner.train()
    elif cfg.mode[:8] == 'valid':
        if runner.use_deform:
            runner.validate_all_mesh(world_space=False, resolution=512, threshold=cfg.mcube_threshold)
            runner.validate_all_image(resolution_level=1)
        else:
            runner.validate_mesh(world_space=False, resolution=512, threshold=cfg.mcube_threshold)
            runner.validate_all_image(resolution_level=1)
    elif cfg.mode == 'train_valid':
        runner.train()
        if runner.use_deform:
            runner.validate_all_mesh(world_space=False, resolution=512, threshold=cfg.mcube_threshold)
            runner.validate_all_image(resolution_level=1)
        else:
            runner.validate_mesh(world_space=False, resolution=512, threshold=cfg.mcube_threshold)
            runner.validate_all_image(resolution_level=1)


if __name__ == '__main__':
    main()
