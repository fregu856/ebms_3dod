import torch
import torch.nn as nn
import logging
from mmcv.runner import load_checkpoint
from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin, MaskTestMixin
from .. import builder
from mmdet.core import (assign_and_sample, bbox2roi, rbbox2roi, bbox2result, multi_apply, kitti_bbox2results,\
                        tensor2points, delta2rbbox3d, weighted_binary_cross_entropy)
import torch.nn.functional as F

import torch.distributions




################################################################################
import math
def gauss_density_centered(x, std):
    return torch.exp(-0.5*(x / std)**2) / (math.sqrt(2*math.pi)*std)

def gmm_density_centered(x, std):
    """
    Assumes dim=-1 is the component dimension and dim=-2 is feature dimension. Rest are sample dimension.
    """
    if x.dim() == std.dim() - 1:
        x = x.unsqueeze(-1)
    elif not (x.dim() == std.dim() and x.shape[-1] == 1):
        raise ValueError('Last dimension must be the gmm stds.')
    return gauss_density_centered(x, std).prod(-2).mean(-1)

def sample_gmm_centered(std, num_samples=1):
    num_components = std.shape[-1]
    num_dims = std.numel() // num_components

    std = std.view(1, num_dims, num_components)

    # Sample component ids
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0,:,k].t()

    # Sample
    x_centered = std_samp * torch.randn(num_samples, num_dims)
    prob_dens = gmm_density_centered(x_centered, std)

    prob_dens_zero = gmm_density_centered(torch.zeros_like(x_centered), std)

    return x_centered, prob_dens, prob_dens_zero

def sample_gmm_centered2(beta, std, num_samples=1):
    num_components = std.shape[-1]
    num_dims = std.numel() // num_components

    std = std.view(1, num_dims, num_components)

    # Sample component ids
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0,:,k].t()

    # Sample
    x_centered = beta*std_samp * torch.randn(num_samples, num_dims)
    prob_dens = gmm_density_centered(x_centered, std)

    prob_dens_zero = gmm_density_centered(torch.zeros_like(x_centered), std)

    return x_centered, prob_dens, prob_dens_zero

stds = torch.zeros((7, 2))
stds[0, 0] = 0.05 # (x)
stds[0, 1] = 0.2 # (x)
stds[1, 0] = 0.05 # (y)
stds[1, 1] = 0.2 # (y)
stds[2, 0] = 0.025 # (z)
stds[2, 1] = 0.1 # (z)
stds[3, 0] = 0.025 # (w)
stds[3, 1] = 0.1 # (w)
stds[4, 0] = 0.025 # (l)
stds[4, 1] = 0.1 # (l)
stds[5, 0] = 0.025 # (h)
stds[5, 1] = 0.1 # (h)
stds[6, 0] = 0.0125 # (theta)
stds[6, 1] = 0.05 # (theta)

stds4 = torch.zeros((7, 3))
stds4[0, 0] = 0.05 # (x)
stds4[0, 1] = 0.1 # (x)
stds4[0, 2] = 0.2 # (x)
stds4[1, 0] = 0.05 # (y)
stds4[1, 1] = 0.1 # (y)
stds4[1, 2] = 0.2 # (y)
stds4[2, 0] = 0.025 # (z)
stds4[2, 1] = 0.05 # (z)
stds4[2, 2] = 0.1 # (z)
stds4[3, 0] = 0.025 # (w)
stds4[3, 1] = 0.05 # (w)
stds4[3, 2] = 0.1 # (w)
stds4[4, 0] = 0.025 # (l)
stds4[4, 1] = 0.05 # (l)
stds4[4, 2] = 0.1 # (l)
stds4[5, 0] = 0.025 # (h)
stds4[5, 1] = 0.05 # (h)
stds4[5, 2] = 0.1 # (h)
stds4[6, 0] = 0.0125 # (theta)
stds4[6, 1] = 0.025 # (theta)
stds4[6, 2] = 0.05 # (theta)

stds8 = 1.25*stds4

import torch.nn.functional as F
min_hwl = 0.05

import numpy as np
def wrapToPi(a):
    return (a + np.pi) % (2*np.pi) - np.pi
################################################################################




################################################################################
from functools import partial

def gen_sample_grid(box, window_size=(4, 7), grid_offsets=(0, 0), spatial_scale=1.):
    # (this function is taken directly from models/single_stage_heads/ssd_rotate_head.py)

    # print (spatial_scale)

    N = box.shape[0]
    win = window_size[0] * window_size[1]
    xg, yg, wg, lg, rg = torch.split(box, 1, dim=-1)

    xg = xg.unsqueeze_(-1).expand(N, *window_size)
    yg = yg.unsqueeze_(-1).expand(N, *window_size)
    rg = rg.unsqueeze_(-1).expand(N, *window_size)

    cosTheta = torch.cos(rg)
    sinTheta = torch.sin(rg)

    xx = torch.linspace(-.5, .5, window_size[0]).type_as(box).view(1, -1) * wg
    yy = torch.linspace(-.5, .5, window_size[1]).type_as(box).view(1, -1) * lg

    xx = xx.unsqueeze_(-1).expand(N, *window_size)
    yy = yy.unsqueeze_(1).expand(N, *window_size)

    x=(xx * cosTheta + yy * sinTheta + xg)
    y=(yy * cosTheta - xx * sinTheta + yg)

    x = (x.permute(1, 2, 0).contiguous() + grid_offsets[0]) * spatial_scale
    y = (y.permute(1, 2, 0).contiguous() + grid_offsets[1]) * spatial_scale

    # return x.view(win, -1), y.view(win, -1)
    return x, y

def bilinear_interpolate_torch_gridsample(input, grid):
    # (input has shape: (N, C, H_in, W_in))
    # (grid has shape: (N, H_out, W_out, 2))

    N, C, H, W = input.shape

    grid[:, :, :, 0] = (grid[:, :, :, 0] / (W - 1))  # normalize to between  0 and 1
    grid[:, :, :, 1] = (grid[:, :, :, 1] / (H - 1))  # normalize to between  0 and 1
    grid = grid * 2 - 1  # normalize to between -1 and 1

    return torch.nn.functional.grid_sample(input, grid)

def make_fc(dim_in, hidden_dim):
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc
################################################################################




class SingleStageDetector(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 extra_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 full_cfg=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if full_cfg is not None:
            if full_cfg.SA_SSD_fixed:
                for p in self.backbone.parameters():
                    p.requires_grad = False

        if neck is not None:
            self.neck = builder.build_neck(neck)

            if full_cfg is not None:
                if full_cfg.SA_SSD_fixed:
                    for p in self.neck.parameters():
                        p.requires_grad = False
        else:
            raise NotImplementedError

        if bbox_head is not None:
            self.rpn_head = builder.build_single_stage_head(bbox_head)

            if full_cfg is not None:
                if full_cfg.SA_SSD_fixed:
                    for p in self.rpn_head.parameters():
                        p.requires_grad = False

        if extra_head is not None:
            self.extra_head = builder.build_single_stage_head(extra_head)

            self.grid_offsets = self.extra_head.grid_offsets
            self.featmap_stride = self.extra_head.featmap_stride
            self.gen_grid_fn = partial(gen_sample_grid, grid_offsets=self.grid_offsets, spatial_scale=(1.0/self.featmap_stride))

            if full_cfg is not None:
                if full_cfg.SA_SSD_fixed:
                    for p in self.extra_head.parameters():
                        p.requires_grad = False

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.full_cfg = full_cfg

        self.num_samples = 8

        self.init_weights(pretrained)

        if full_cfg is not None:
            if full_cfg.USE_EBM:
                self.ebm_fc1 = make_fc(7168, 1024)
                self.ebm_fc2 = make_fc(1024, 1024)
                self.ebm_fc3 = nn.Linear(1024, 1)
                nn.init.normal_(self.ebm_fc3.weight, std=0.001)
                for l in [self.ebm_fc3]:
                    nn.init.constant_(l.bias, 0)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def merge_second_batch(self, batch_args):
        ret = {}
        for key, elems in batch_args.items():
            if key in [
                'voxels', 'num_points',
            ]:
                ret[key] = torch.cat(elems, dim=0)
            elif key == 'coordinates':
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = F.pad(
                        coor, [1, 0, 0, 0],
                        mode='constant',
                        value=i)
                    coors.append(coor_pad)
                ret[key] = torch.cat(coors, dim=0)
            elif key in [
                'img_meta', 'gt_labels', 'gt_bboxes',
            ]:
                ret[key] = elems
            else:
                ret[key] = torch.stack(elems, dim=0)
        return ret

    def forward_train(self, img, img_meta, **kwargs):
        # (img has shape: (batch_size, 3, 384, 1248))
        # (img_meta is a list of batch_size elements, example element: {'img_shape': (375, 1242, 3), 'sample_idx': 3132, 'calib': <mmdet.datasets.kitti_utils.Calibration object at 0x7fc3c16ad898>})
        # (kwargs is a dict containing the keys "anchors", "voxels", "coordinates", "num_points", "anchors_mask", "gt_labels", "gt_bboxes")
        # # (kwargs["anchors"] etc is a list of batch_size tensors)

        # print (img.size())
        # print (len(img_meta))

        batch_size = len(img_meta)

        ret = self.merge_second_batch(kwargs)
        # (ret["voxels"] has shape: (num_voxels, 4)) (num_voxels is different for different examples) (for batch_size = 2, num_voxels is typically 35000 - 45000)
        # (ret["coordinates"] has shape: (num_voxels, 4))
        # print (ret["voxels"].size())
        # print (ret["coordinates"].size())

        vx = self.backbone(ret['voxels'], ret['num_points'])
        # (vx has shape: (num_voxels, 4)) (vx is just identical to ret["voxels"]? seems so)
        # print (vx.size())

        (x, conv6), point_misc = self.neck(vx, ret['coordinates'], batch_size)
        # (x has shape: (batch_size, 256, 200, 176))
        # (conv6 has shape: (batch_size, 256, 200, 176))
        # (point_misc is a list of 3 tensors)
        # print (x.size())
        # print (conv6.size())

        losses = dict()

        if not self.full_cfg.SA_SSD_fixed:
            aux_loss = self.neck.aux_loss(*point_misc, gt_bboxes=ret['gt_bboxes'])
            losses.update(aux_loss)

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            # (rpn_outs is a list of 3 elements)
            # (rpn_outs[0] has shape: (batch_size, 200, 176, 14)) (14 = 7*num_anchor_per_loc) (x, y, z, h, w, l, theta)
            # (rpn_outs[1] has shape: (batch_size, 200, 176, 2)) (2 = 1*num_anchor_per_loc) (conf_score) (just one class (Car))
            # (rpn_outs[2] has shape: (batch_size, 200, 176, 4)) (4 = 2*num_anchor_per_loc) (classification of heading directon (forward or backward))
            # print (len(rpn_outs))
            # print (rpn_outs[0].size())
            # print (rpn_outs[1].size())
            # print (rpn_outs[2].size())

            if not self.full_cfg.SA_SSD_fixed:
                rpn_loss_inputs = rpn_outs + (ret['gt_bboxes'], ret['gt_labels'], ret['anchors'], ret['anchors_mask'], self.train_cfg.rpn)
                rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
                losses.update(rpn_losses)

            guided_anchors = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'], ret['gt_bboxes'], thr=0.1)
            # (guided_anchors is a list of batch_size tensors)
            # (guided_anchors[i] has shape: (num_guided_anchors_in_pc_i, 7))
            # # (num_guided_anchors_in_pc_i is different for different i:s and for different examples) (typically, num_guided_anchors_in_pc_i is ~ 10000 - 25000)
            # # (these are the predicted bboxes (with residuals added to the anchors) with conf_score > 0.1?)
            # print (len(guided_anchors))
            # print (guided_anchors[0].size())
            # print (guided_anchors[1].size())
        else:
            raise NotImplementedError

        # bbox head forward and loss
        if self.extra_head:
            bbox_score = self.extra_head(conv6, guided_anchors)
            # print (bbox_score.size())
            # (bbox_score has shape: (num_guided_anchors_in_batch))
            # # (num_guided_anchors_in_batch = num_guided_anchors_in_pc_0 + num_guided_anchors_in_pc_1 + ... + num_guided_anchors_in_pc_{batch_size - 1})

            if not self.full_cfg.SA_SSD_fixed:
                refine_loss_inputs = (bbox_score, ret['gt_bboxes'], ret['gt_labels'], guided_anchors, self.train_cfg.extra)
                refine_losses = self.extra_head.loss(*refine_loss_inputs)
                losses.update(refine_losses)

        if self.full_cfg is not None:
            if self.full_cfg.USE_EBM:
                # print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                # (conv6 has shape: (batch_size, 256, 200, 176))

                # (ret["gt_bboxes"] is a list of batch_size tensors)
                # (ret["gt_bboxes"][i] has shape: (num_gt_bboxes_i, 7)) (num_gt_bboxes_i can be different for different i:s and for different batches)

                print (conv6.size())
                print (len(ret['gt_bboxes']))
                print (ret['gt_bboxes'][0].size())
                print (ret['gt_bboxes'][1].size())

                batch_size = len(ret['gt_bboxes'])

                ys_list = ret["gt_bboxes"]
                y_samples_list = []
                q_y_samples_list = []
                q_ys_list = []
                for i in range(batch_size):
                    # (ys_list[i] has shape: (num_gt_bboxes_i, 7))
                    # print (ys_list[i].size())

                    y_samples_zero, q_y_samples, q_ys = sample_gmm_centered(stds, num_samples=self.num_samples)
                    y_samples_zero = y_samples_zero.cuda() # (shape: (num_samples, 7))
                    q_y_samples = q_y_samples.cuda() # (shape: (num_samples))
                    y_samples = ys_list[i].unsqueeze(1) + y_samples_zero.unsqueeze(0) # (shape: (num_gt_bboxes_i, num_samples, 7))
                    y_samples[:, :, 3:6] = min_hwl + F.relu(y_samples[:, :, 3:6] - min_hwl)
                    y_samples[:, :, 6] = wrapToPi(y_samples[:, :, 6])
                    q_y_samples = q_y_samples.unsqueeze(0)*torch.ones(y_samples.size(0), y_samples.size(1)).cuda() # (shape: (num_gt_bboxes_i, num_samples))
                    q_ys = q_ys[0]*torch.ones(ys_list[i].size(0)).cuda() # (shape: (num_gt_bboxes_i))

                    # print (ys_list[i][0])
                    # print (y_samples_list[i][0, 0:5])

                    y_samples = y_samples.view(-1, 7) # (shape: (num_gt_bboxes_i*num_samples, 7)))
                    q_y_samples = q_y_samples.view(-1) # (shape: (num_gt_bboxes_i*num_samples)))

                    y_samples_list.append(y_samples)
                    q_y_samples_list.append(q_y_samples)
                    q_ys_list.append(q_ys)

                    # print (y_samples_list[i].size())
                    # print (q_y_samples_list[i].size())
                    # print (q_ys_list[i].size())
                    # print ("%%%%%")

                ys_features_list = []
                y_samples_features_list = []
                for i in range(batch_size):
                    # (conv6 has shape: (batch_size, 256, 200, 176))
                    # (ys_list[i] has shape: (num_gt_bboxes_i, 7))
                    # (y_samples_list[i] has shape: (num_gt_bboxes_i*num_samples, 7))
                    # print (conv6.size())
                    # print (ys_list[i].size())
                    # print (y_samples_list[i].size())

                    (ys_pixel_xs, ys_pixel_ys) = self.gen_grid_fn(ys_list[i][:, [0, 1, 3, 4, 6]])
                    # (both have shape: (4, 7, num_gt_bboxes_i))
                    # print (ys_pixel_xs.size())
                    # print (ys_pixel_ys.size())
                    ys_pixel_xs = ys_pixel_xs.permute(2, 0, 1).contiguous() # (shape: (num_gt_bboxes_i, 4, 7))
                    ys_pixel_ys = ys_pixel_ys.permute(2, 0, 1).contiguous() # (shape: (num_gt_bboxes_i, 4, 7))
                    # print (ys_pixel_xs.size())
                    # print (ys_pixel_ys.size())
                    ys_pixel_coords = torch.cat([ys_pixel_xs.unsqueeze(3), ys_pixel_ys.unsqueeze(3)], 3)
                    # (shape: (num_gt_bboxes_i, 4, 7, 2))
                    # print (ys_pixel_coords.size())

                    (y_samples_pixel_xs, y_samples_pixel_ys) = self.gen_grid_fn(y_samples_list[i][:, [0, 1, 3, 4, 6]])
                    # (both have shape: (4, 7, num_gt_bboxes_i*num_samples))
                    # print (y_samples_pixel_xs.size())
                    # print (y_samples_pixel_ys.size())
                    y_samples_pixel_xs = y_samples_pixel_xs.permute(2, 0, 1).contiguous() # (shape: (num_gt_bboxes_i*num_samples, 4, 7))
                    y_samples_pixel_ys = y_samples_pixel_ys.permute(2, 0, 1).contiguous() # (shape: (num_gt_bboxes_i*num_samples, 4, 7))
                    # print (y_samples_pixel_xs.size())
                    # print (y_samples_pixel_ys.size())
                    y_samples_pixel_coords = torch.cat([y_samples_pixel_xs.unsqueeze(3), y_samples_pixel_ys.unsqueeze(3)], 3)
                    # (shape: (num_gt_bboxes_i*num_samples, 4, 7, 2))
                    # print (y_samples_pixel_coords.size())

                    conv6_i = conv6[i].unsqueeze(0) # (shape: (1, 256, 200, 176))
                    # print (conv6_i.size())
                    conv6_i_ys = conv6_i.expand(ys_pixel_coords.size(0), -1, -1, -1)
                    # (shape: (num_gt_bboxes_i, 256, 200, 176))
                    # print (conv6_i_ys.size())
                    ys_feature_maps = bilinear_interpolate_torch_gridsample(conv6_i_ys, ys_pixel_coords)
                    # (shape: (num_gt_bboxes_i, 256, 4, 7))
                    # print (ys_feature_maps.size())
                    ys_features = ys_feature_maps.view(ys_feature_maps.size(0), -1)
                    # (shape: (num_gt_bboxes_i, 7168)) (7168 = 256*4*7)
                    # print (ys_features.size())
                    ys_features_list.append(ys_features)

                    conv6_i_y_samples = conv6_i.expand(y_samples_pixel_coords.size(0), -1, -1, -1)
                    # (shape: (num_gt_bboxes_i*num_samples, 256, 200, 176))
                    # print (conv6_i_y_samples.size())
                    y_samples_feature_maps = bilinear_interpolate_torch_gridsample(conv6_i_y_samples, y_samples_pixel_coords)
                    # (shape: (num_gt_bboxes_i*num_samples, 256, 4, 7))
                    # print (y_samples_feature_maps.size())
                    y_samples_features = y_samples_feature_maps.view(y_samples_feature_maps.size(0), -1)
                    # (shape: (num_gt_bboxes_i*num_samples, 7168)) (7168 = 256*4*7)
                    # print (y_samples_features.size())
                    y_samples_features_list.append(y_samples_features)

                # print (ys_features_list[0].size())
                # print (ys_features_list[1].size())
                ys_features = torch.cat(ys_features_list, 0)
                # (shape: (num_gt_bboxes_in_batch, 7168))
                # print (ys_features.size())

                # print (y_samples_features_list[0].size())
                # print (y_samples_features_list[1].size())
                y_samples_features = torch.cat(y_samples_features_list, 0)
                # (shape: (num_gt_bboxes_in_batch*num_samples, 7168))
                # print (y_samples_features.size())

                features = torch.cat([ys_features, y_samples_features], 0)
                # (shape: (num_gt_bboxes_in_batch + num_gt_bboxes_in_batch*num_samples, 7168))
                # print (features.size())

                features = F.relu(self.ebm_fc1(features)) # (shape: (num_gt_bboxes_in_batch + num_gt_bboxes_in_batch*num_samples, 1024))
                # print (features.size())
                features = F.relu(self.ebm_fc2(features)) # (shape: (num_gt_bboxes_in_batch + num_gt_bboxes_in_batch*num_samples, 1024))
                # print (features.size())

                fs = self.ebm_fc3(features) # (shape: (num_gt_bboxes_in_batch + num_gt_bboxes_in_batch*num_samples, 1))
                # print (fs.size())
                fs = fs.squeeze(1) # (shape: (num_gt_bboxes_in_batch + num_gt_bboxes_in_batch*num_samples))
                # print (fs.size())

                ys_fs = fs[0:ys_features.size(0)]
                # (shape: (num_gt_bboxes_in_batch))
                # print (ys_fs.size())

                y_samples_fs = fs[ys_features.size(0):]
                # (shape: (num_gt_bboxes_in_batch*num_samples))
                # print (y_samples_fs.size())

                y_samples_fs = y_samples_fs.view(-1, self.num_samples)
                # (shape: (num_gt_bboxes_in_batch, num_samples))
                # print (y_samples_fs.size())

                q_ys = torch.cat(q_ys_list, 0)
                # (shape: (num_gt_bboxes_in_batch))
                # print (q_ys.size())

                q_y_samples = torch.cat(q_y_samples_list, 0)
                # (shape: (num_gt_bboxes_in_batch*num_samples))
                # print (q_y_samples.size())

                q_y_samples = q_y_samples.view(-1, self.num_samples)
                # (shape: (num_gt_bboxes_in_batch, num_samples))
                # print (q_y_samples.size())

                # print ("//////////////////")
                # (ys_fs has shape: (num_gt_bboxes_in_batch))
                # (y_samples_fs has shape: (num_gt_bboxes_in_batch, num_samples))
                # (q_ys has shape: (num_gt_bboxes_in_batch))
                # (q_y_samples has shape: (num_gt_bboxes_in_batch, num_samples))

                print (ys_fs.size())
                print (y_samples_fs.size())
                print (q_ys.size())
                print (q_y_samples.size())

                # print (ys_fs[0])
                # print (y_samples_fs[0])
                # print (q_ys)
                # print (q_y_samples[0])

                print ("ys_fs - mean: %f, max: %f, min: %f" % (torch.mean(ys_fs).item(), torch.max(ys_fs).item(), torch.min(ys_fs).item()))
                print ("y_samples_fs - mean: %f, max: %f, min: %f" % (torch.mean(y_samples_fs).item(), torch.max(y_samples_fs).item(), torch.min(y_samples_fs).item()))

                f_samples = y_samples_fs # (shape: (num_gt_bboxes_in_batch, num_samples))
                p_N_samples = q_y_samples # (shape: (num_gt_bboxes_in_batch, num_samples))
                f_0 = ys_fs # (shape: (num_gt_bboxes_in_batch))
                p_N_0 = q_ys # (shape: (num_gt_bboxes_in_batch))
                exp_vals_0 = f_0-torch.log(p_N_0 + 0.0) # (shape: (num_gt_bboxes_in_batch))
                exp_vals_samples = f_samples-torch.log(p_N_samples + 0.0) # (shape: (num_gt_bboxes_in_batch, num_samples))
                exp_vals = torch.cat([exp_vals_0.unsqueeze(1), exp_vals_samples], dim=1) # (shape: (num_gt_bboxes_in_batch, 1+num_samples))
                ebm_loss = -torch.mean(exp_vals_0 - torch.logsumexp(exp_vals, dim=1))
                losses.update(dict(loss_ebm=ebm_loss,))

        print ("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")

        return losses

    def forward_test(self, img, img_meta, **kwargs):
        with torch.no_grad():
            batch_size = len(img_meta) # (batch_size = 1)
            # print (batch_size)

            ret = self.merge_second_batch(kwargs)
            # (ret["voxels"] has shape: (num_voxels, 4)) (num_voxels is different for different examples) (for batch_size = 2, num_voxels is typically 35000 - 45000)
            # (ret["coordinates"] has shape: (num_voxels, 4))
            # print (ret["voxels"].size())
            # print (ret["coordinates"].size())

            vx = self.backbone(ret['voxels'], ret['num_points'])
            # (vx has shape: (num_voxels, 4)) (vx is just identical to ret["voxels"]? seems so)
            # print (vx.size())

            (x, conv6) = self.neck(vx, ret['coordinates'], batch_size, is_test=True)
            # (x has shape: (batch_size, 256, 200, 176))
            # (conv6 has shape: (batch_size, 256, 200, 176))
            # print (x.size())
            # print (conv6.size())

            rpn_outs = self.rpn_head.forward(x)
            # (rpn_outs is a list of 3 elements)
            # (rpn_outs[0] has shape: (batch_size, 200, 176, 14)) (14 = 7*num_anchor_per_loc) (x, y, z, h, w, l, theta)
            # (rpn_outs[1] has shape: (batch_size, 200, 176, 2)) (2 = 1*num_anchor_per_loc) (conf_score) (just one class (Car))
            # (rpn_outs[2] has shape: (batch_size, 200, 176, 4)) (4 = 2*num_anchor_per_loc) (classification of heading directon (forward or backward))
            # print (len(rpn_outs))
            # print (rpn_outs[0].size())
            # print (rpn_outs[1].size())
            # print (rpn_outs[2].size())

            guided_anchors = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'], None, thr=.1)
            # (guided_anchors is a list of batch_size tensors)
            # (guided_anchors[i] has shape: (num_guided_anchors_in_pc_i, 7))
            # # (num_guided_anchors_in_pc_i is different for different i:s and for different examples)
            # # (these are the predicted bboxes (with residuals added to the anchors) with conf_score > 0.1?)
            # print (len(guided_anchors))
            # print (guided_anchors[0].size())

            bbox_score, guided_anchors = self.extra_head(conv6, guided_anchors, is_test=True)
            # (bbox_score is a list of batch_size tensors)
            # # (bbox_score[i] has shape: (num_guided_anchors_in_pc_i))
            # (guided_anchors is a list of batch_size tensors)
            # # (guided_anchors[i] has shape: (num_guided_anchors_in_pc_i, 7))
            # print (len(bbox_score))
            # print (bbox_score[0].size())
            # print (bbox_score[0])
            # print (len(guided_anchors))
            # print (guided_anchors[0].size())

            if self.full_cfg is not None:
                if self.full_cfg.USE_EBM and (self.test_cfg.extra.EBM_guided or self.test_cfg.extra.EBM_refine):
                    # print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    # (conv6 has shape: (batch_size, 256, 200, 176))
                    # (guided_anchors is a list of batch_size tensors)
                    # (guided_anchors[i] has shape: (num_guided_anchors_in_pc_i, 7))

                    batch_size = len(guided_anchors)

                    ys_list = guided_anchors

                    ys_features_list = []
                    for i in range(batch_size):
                        # (conv6 has shape: (batch_size, 256, 200, 176))
                        # (ys_list[i] has shape: (num_guided_anchors_in_pc_i, 7))
                        # print (conv6.size())
                        # print (ys_list[i].size())

                        (ys_pixel_xs, ys_pixel_ys) = self.gen_grid_fn(ys_list[i][:, [0, 1, 3, 4, 6]])
                        # (both have shape: (4, 7, num_guided_anchors_in_pc_i))
                        # print (ys_pixel_xs.size())
                        # print (ys_pixel_ys.size())
                        ys_pixel_xs = ys_pixel_xs.permute(2, 0, 1).contiguous() # (shape: (num_guided_anchors_in_pc_i, 4, 7))
                        ys_pixel_ys = ys_pixel_ys.permute(2, 0, 1).contiguous() # (shape: (num_guided_anchors_in_pc_i, 4, 7))
                        # print (ys_pixel_xs.size())
                        # print (ys_pixel_ys.size())
                        ys_pixel_coords = torch.cat([ys_pixel_xs.unsqueeze(3), ys_pixel_ys.unsqueeze(3)], 3)
                        # (shape: (num_guided_anchors_in_pc_i, 4, 7, 2))
                        # print (ys_pixel_coords.size())

                        conv6_i = conv6[i].unsqueeze(0) # (shape: (1, 256, 200, 176))
                        # print (conv6_i.size())
                        conv6_i_ys = conv6_i.expand(ys_pixel_coords.size(0), -1, -1, -1)
                        # (shape: (num_guided_anchors_in_pc_i, 256, 200, 176))
                        # print (conv6_i_ys.size())

                        if conv6_i_ys.size(0) < 150:
                            ys_feature_maps = bilinear_interpolate_torch_gridsample(conv6_i_ys, ys_pixel_coords)
                            # (shape: (num_guided_anchors_in_pc_i, 256, 4, 7))
                            # print (ys_feature_maps.size())
                        else:
                            num_iters = int(math.floor(conv6_i_ys.size(0)/150.0))
                            ys_feature_maps_list = []
                            for iter in range(num_iters):
                                ys_feature_maps_list.append(bilinear_interpolate_torch_gridsample(conv6_i_ys[(150*iter):(150*(iter+1))], ys_pixel_coords[(150*iter):(150*(iter+1))]))
                            ys_feature_maps_list.append(bilinear_interpolate_torch_gridsample(conv6_i_ys[(150*num_iters):], ys_pixel_coords[(150*num_iters):]))
                            ys_feature_maps = torch.cat(ys_feature_maps_list, 0)
                            # (shape: (num_guided_anchors_in_pc_i, 256, 4, 7))

                        ys_features = ys_feature_maps.view(ys_feature_maps.size(0), -1)
                        # (shape: (num_guided_anchors_in_pc_i, 7168)) (7168 = 256*4*7)
                        # print (ys_features.size())
                        ys_features_list.append(ys_features)

                    fs_list = []
                    for i in range(batch_size):
                        features = F.relu(self.ebm_fc1(ys_features_list[i])) # (shape: (num_guided_anchors_in_pc_i, 1024))
                        # print (features.size())
                        features = F.relu(self.ebm_fc2(features)) # (shape: (num_guided_anchors_in_pc_i, 1024))
                        # print (features.size())

                        fs = self.ebm_fc3(features) # (shape: (num_guided_anchors_in_pc_i, 1))
                        # print (fs.size())
                        fs = fs.squeeze(1) # (shape: (num_guided_anchors_in_pc_i))
                        # print (fs.size())

                        fs_list.append(fs)

                # (fs_list is a list of batch_size tensors)
                # # (fs_list[i] has shape: (num_guided_anchors_in_pc_i))
                # print (len(fs_list))
                # print (fs_list[0].size())
                # print (fs_list[0])

            if self.test_cfg.extra.EBM_guided:
                det_bboxes, det_scores, det_fs = self.extra_head.get_rescore_bboxes_ebm_guided(
                    guided_anchors, bbox_score, fs_list, img_meta, self.test_cfg.extra)
            else:
                det_bboxes, det_scores = self.extra_head.get_rescore_bboxes(
                    guided_anchors, bbox_score, img_meta, self.test_cfg.extra)
            # (det_scores is a list of batch_size numpy arrays)
            # # (det_scores[i] has shape: (num_detections_i)) (num_detections_i <= num_guided_anchors_in_pc_i)
            # (det_fs is a list of batch_size numpy arrays)
            # # (det_fs[i] has shape: (num_detections_i))
            # (det_bboxes is a list of batch_size numpy arrays)
            # # (det_bboxes[i] has shape: (num_detections_i, 7))
            # print (len(det_scores))
            # print (det_scores[0].shape)
            # print (len(det_bboxes))
            # print (det_bboxes[0].shape)

            # print (" ")
            # print ("fs before refinement:")
            # print (det_fs.detach().cpu().numpy())
            # print ("bboxes before refinement:")
            # print (det_bboxes[0])
            # print ("%%%%%%%%%%%%%%%%%%%%%%")
        # (end of "with torch.no_grad():"") ####################################

        if self.test_cfg.extra.EBM_refine:
            # (det_bboxes is a list of batch_size numpy arrays)
            # # (det_bboxes[i] has shape: (num_detections_i, 7))
            # (conv6 has shape: (batch_size, 256, 200, 176))
            # (batch_size == 1)
            # print (conv6.size())

            bboxes = []
            for i in range(len(det_bboxes)):
                bboxes.append(torch.from_numpy(det_bboxes[i]).cuda())
            # (bboxes is a list of batch_size tensors)
            # # (bboxes[i] has shape: (num_detections_i, 7))
            # print (len(bboxes))
            # print (bboxes[0].size())

            conv6.requires_grad = True

            det_bboxes = []
            for i in range(len(bboxes)):
                # (conv6 has shape: (batch_size, 256, 200, 176))
                # (bboxes[i] has shape: (num_detections_i, 7))
                # print (conv6.size())
                # print (bboxes[i].size())

                bboxes_i = bboxes[i] # (shape: (num_detections_i, 7))
                if bboxes_i.size(0) == 0:
                    det_bboxes.append(bboxes_i.detach().cpu().numpy())
                    continue

                step_sizes = 0.0001*torch.ones(bboxes_i.size(0), 1).cuda() # (shape: (num_detections_i, 1))
                print (self.test_cfg.extra.EBM_refine_steps)
                for step in range(self.test_cfg.extra.EBM_refine_steps):
                    # print (step_sizes)

                    bboxes_init = bboxes_i.clone().detach() # (shape: (num_detections_i, 7))

                    bboxes_init.requires_grad = True

                    # print (bboxes_init[0])
                    # print (bboxes_init.size())

                    ############################################################
                    ############################################################
                    ############################################################
                    ############################################################
                    ############################################################
                    (ys_pixel_xs, ys_pixel_ys) = self.gen_grid_fn(bboxes_init[:, [0, 1, 3, 4, 6]])
                    # (both have shape: (4, 7, num_detections_i))
                    # print (ys_pixel_xs.size())
                    # print (ys_pixel_ys.size())
                    ys_pixel_xs = ys_pixel_xs.permute(2, 0, 1).contiguous() # (shape: (num_detections_i, 4, 7))
                    ys_pixel_ys = ys_pixel_ys.permute(2, 0, 1).contiguous() # (shape: (num_detections_i, 4, 7))
                    # print (ys_pixel_xs.size())
                    # print (ys_pixel_ys.size())
                    ys_pixel_coords = torch.cat([ys_pixel_xs.unsqueeze(3), ys_pixel_ys.unsqueeze(3)], 3)
                    # (shape: (num_detections_i, 4, 7, 2))
                    # print (ys_pixel_coords.size())
                    #
                    conv6_i = conv6[i].unsqueeze(0) # (shape: (1, 256, 200, 176))
                    # print (conv6_i.size())
                    conv6_i_ys = conv6_i.expand(ys_pixel_coords.size(0), -1, -1, -1)
                    # (shape: (num_detections_i, 256, 200, 176))
                    # print (conv6_i_ys.size())
                    #
                    if conv6_i_ys.size(0) < 150:
                        ys_feature_maps = bilinear_interpolate_torch_gridsample(conv6_i_ys, ys_pixel_coords)
                        # (shape: (num_detections_i, 256, 4, 7))
                        # print (ys_feature_maps.size())
                    else:
                        num_iters = int(math.floor(conv6_i_ys.size(0)/150.0))
                        ys_feature_maps_list = []
                        for iter in range(num_iters):
                            ys_feature_maps_list.append(bilinear_interpolate_torch_gridsample(conv6_i_ys[(150*iter):(150*(iter+1))], ys_pixel_coords[(150*iter):(150*(iter+1))]))
                        ys_feature_maps_list.append(bilinear_interpolate_torch_gridsample(conv6_i_ys[(150*num_iters):], ys_pixel_coords[(150*num_iters):]))
                        ys_feature_maps = torch.cat(ys_feature_maps_list, 0)
                        # (shape: (num_detections_i, 256, 4, 7))
                    #
                    ys_features = ys_feature_maps.view(ys_feature_maps.size(0), -1)
                    # (shape: (num_detections_i, 7168)) (7168 = 256*4*7)
                    # print (ys_features.size())
                    #
                    features = F.relu(self.ebm_fc1(ys_features)) # (shape: (num_detections_i, 1024))
                    # print (features.size())
                    features = F.relu(self.ebm_fc2(features)) # (shape: (num_detections_i, 1024))
                    # print (features.size())
                    #
                    fs = self.ebm_fc3(features) # (shape: (num_detections_i, 1))
                    # print (fs.size())
                    fs = fs.squeeze(1) # (shape: (num_detections_i))
                    # print (fs.size())
                    # print (fs)
                    ############################################################
                    ############################################################
                    ############################################################
                    ############################################################
                    ############################################################

                    # fs.backward(gradient = torch.ones_like(fs))
                    #
                    grad_bboxes_init = torch.autograd.grad(fs.sum(), bboxes_init, create_graph=True)[0]
                    # (shape: (num_detections_i, 7)) (same as bboxes_init)
                    # print (grad_bboxes_init.size())

                    # bboxes_refined = bboxes_init + 0.0001*bboxes_init.grad
                    #
                    bboxes_refined = bboxes_init + step_sizes*grad_bboxes_init

                    with torch.no_grad():
                        ############################################################
                        ############################################################
                        ############################################################
                        ############################################################
                        ############################################################
                        (ys_pixel_xs, ys_pixel_ys) = self.gen_grid_fn(bboxes_refined[:, [0, 1, 3, 4, 6]])
                        # (both have shape: (4, 7, num_detections_i))
                        # print (ys_pixel_xs.size())
                        # print (ys_pixel_ys.size())
                        ys_pixel_xs = ys_pixel_xs.permute(2, 0, 1).contiguous() # (shape: (num_detections_i, 4, 7))
                        ys_pixel_ys = ys_pixel_ys.permute(2, 0, 1).contiguous() # (shape: (num_detections_i, 4, 7))
                        # print (ys_pixel_xs.size())
                        # print (ys_pixel_ys.size())
                        ys_pixel_coords = torch.cat([ys_pixel_xs.unsqueeze(3), ys_pixel_ys.unsqueeze(3)], 3)
                        # (shape: (num_detections_i, 4, 7, 2))
                        # print (ys_pixel_coords.size())
                        #
                        conv6_i = conv6[i].unsqueeze(0) # (shape: (1, 256, 200, 176))
                        # print (conv6_i.size())
                        conv6_i_ys = conv6_i.expand(ys_pixel_coords.size(0), -1, -1, -1)
                        # (shape: (num_detections_i, 256, 200, 176))
                        # print (conv6_i_ys.size())
                        #
                        if conv6_i_ys.size(0) < 150:
                            ys_feature_maps = bilinear_interpolate_torch_gridsample(conv6_i_ys, ys_pixel_coords)
                            # (shape: (num_detections_i, 256, 4, 7))
                            # print (ys_feature_maps.size())
                        else:
                            num_iters = int(math.floor(conv6_i_ys.size(0)/150.0))
                            ys_feature_maps_list = []
                            for iter in range(num_iters):
                                ys_feature_maps_list.append(bilinear_interpolate_torch_gridsample(conv6_i_ys[(150*iter):(150*(iter+1))], ys_pixel_coords[(150*iter):(150*(iter+1))]))
                            ys_feature_maps_list.append(bilinear_interpolate_torch_gridsample(conv6_i_ys[(150*num_iters):], ys_pixel_coords[(150*num_iters):]))
                            ys_feature_maps = torch.cat(ys_feature_maps_list, 0)
                            # (shape: (num_detections_i, 256, 4, 7))
                        #
                        ys_features = ys_feature_maps.view(ys_feature_maps.size(0), -1)
                        # (shape: (num_detections_i, 7168)) (7168 = 256*4*7)
                        # print (ys_features.size())
                        #
                        features = F.relu(self.ebm_fc1(ys_features)) # (shape: (num_detections_i, 1024))
                        # print (features.size())
                        features = F.relu(self.ebm_fc2(features)) # (shape: (num_detections_i, 1024))
                        # print (features.size())
                        #
                        new_fs = self.ebm_fc3(features) # (shape: (num_detections_i, 1))
                        # print (new_fs.size())
                        new_fs = new_fs.squeeze(1) # (shape: (num_detections_i))
                        # print (new_fs.size())
                        # print (new_fs)
                        ############################################################
                        ############################################################
                        ############################################################
                        ############################################################
                        ############################################################

                        refinement_failed = (new_fs < fs) # (shape: (num_detections_i))
                        # print (refinement_failed)
                        # print (refinement_failed.size())
                        refinement_failed = refinement_failed.unsqueeze(1) # (shape: (num_detections_i, 1))
                        r_f = refinement_failed.float()

                        bboxes_i = r_f*bboxes_init + (1.0-r_f)*bboxes_refined

                        step_sizes = (1.0-r_f)*step_sizes + r_f*0.5*step_sizes

                        if step == self.test_cfg.extra.EBM_refine_steps - 1: # (in final step)
                            refinement_failed = (new_fs < fs) # (shape: (num_detections_i))
                            # print (refinement_failed)
                            # print (refinement_failed.size())
                            r_f = refinement_failed.float()
                            final_fs = r_f*fs + (1.0-r_f)*new_fs

                            # print ("###")
                            # print ("###")
                            # print ("###")
                            # print ("fs after refinement:")
                            # print (final_fs.detach().cpu().numpy())

                    # print ("***********************")

                det_bboxes.append(bboxes_i.detach().cpu().numpy())

        # print ("bboxes after refinement:")
        # print (det_bboxes[0])

        results = [kitti_bbox2results(*param) for param in zip(det_bboxes, det_scores, img_meta)]

        print ("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")

        return results




class SingleStageDetector20(BaseDetector, RPNTestMixin, BBoxTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 extra_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 full_cfg=None):
        super(SingleStageDetector20, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if full_cfg is not None:
            if full_cfg.SA_SSD_fixed:
                for p in self.backbone.parameters():
                    p.requires_grad = False

        if neck is not None:
            self.neck = builder.build_neck(neck)

            if full_cfg is not None:
                if full_cfg.SA_SSD_fixed:
                    for p in self.neck.parameters():
                        p.requires_grad = False
        else:
            raise NotImplementedError

        if bbox_head is not None:
            self.rpn_head = builder.build_single_stage_head(bbox_head)

            if full_cfg is not None:
                if full_cfg.SA_SSD_fixed:
                    for p in self.rpn_head.parameters():
                        p.requires_grad = False

        if extra_head is not None:
            self.extra_head = builder.build_single_stage_head(extra_head)

            self.grid_offsets = self.extra_head.grid_offsets
            self.featmap_stride = self.extra_head.featmap_stride
            self.gen_grid_fn = partial(gen_sample_grid, grid_offsets=self.grid_offsets, spatial_scale=(1.0/self.featmap_stride))

            if full_cfg is not None:
                if full_cfg.SA_SSD_fixed:
                    for p in self.extra_head.parameters():
                        p.requires_grad = False

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.full_cfg = full_cfg

        self.num_samples = 128

        self.stds = stds8
        print (self.stds)

        self.init_weights(pretrained)

        if full_cfg is not None:
            if full_cfg.USE_EBM:
                self.ebm_fc1 = make_fc(7168+16+16, 1024)
                self.ebm_fc2 = make_fc(1024, 1024)
                self.ebm_fc3 = nn.Linear(1024, 1)
                nn.init.normal_(self.ebm_fc3.weight, std=0.001)
                for l in [self.ebm_fc3]:
                    nn.init.constant_(l.bias, 0)

                self.z_fc1 = nn.Linear(1, 16)
                self.z_fc2 = nn.Linear(16, 16)

                self.h_fc1 = nn.Linear(1, 16)
                self.h_fc2 = nn.Linear(16, 16)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def merge_second_batch(self, batch_args):
        ret = {}
        for key, elems in batch_args.items():
            if key in [
                'voxels', 'num_points',
            ]:
                ret[key] = torch.cat(elems, dim=0)
            elif key == 'coordinates':
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = F.pad(
                        coor, [1, 0, 0, 0],
                        mode='constant',
                        value=i)
                    coors.append(coor_pad)
                ret[key] = torch.cat(coors, dim=0)
            elif key in [
                'img_meta', 'gt_labels', 'gt_bboxes',
            ]:
                ret[key] = elems
            else:
                ret[key] = torch.stack(elems, dim=0)
        return ret

    def forward_train(self, img, img_meta, **kwargs):
        # (img has shape: (batch_size, 3, 384, 1248))
        # (img_meta is a list of batch_size elements, example element: {'img_shape': (375, 1242, 3), 'sample_idx': 3132, 'calib': <mmdet.datasets.kitti_utils.Calibration object at 0x7fc3c16ad898>})
        # (kwargs is a dict containing the keys "anchors", "voxels", "coordinates", "num_points", "anchors_mask", "gt_labels", "gt_bboxes")
        # # (kwargs["anchors"] etc is a list of batch_size tensors)

        # print (img.size())
        # print (len(img_meta))

        batch_size = len(img_meta)

        ret = self.merge_second_batch(kwargs)
        # (ret["voxels"] has shape: (num_voxels, 4)) (num_voxels is different for different examples) (for batch_size = 2, num_voxels is typically 35000 - 45000)
        # (ret["coordinates"] has shape: (num_voxels, 4))
        # print (ret["voxels"].size())
        # print (ret["coordinates"].size())

        vx = self.backbone(ret['voxels'], ret['num_points'])
        # (vx has shape: (num_voxels, 4)) (vx is just identical to ret["voxels"]? seems so)
        # print (vx.size())

        (x, conv6), point_misc = self.neck(vx, ret['coordinates'], batch_size)
        # (x has shape: (batch_size, 256, 200, 176))
        # (conv6 has shape: (batch_size, 256, 200, 176))
        # (point_misc is a list of 3 tensors)
        # print (x.size())
        # print (conv6.size())

        losses = dict()

        if not self.full_cfg.SA_SSD_fixed:
            aux_loss = self.neck.aux_loss(*point_misc, gt_bboxes=ret['gt_bboxes'])
            losses.update(aux_loss)

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            # (rpn_outs is a list of 3 elements)
            # (rpn_outs[0] has shape: (batch_size, 200, 176, 14)) (14 = 7*num_anchor_per_loc) (x, y, z, h, w, l, theta)
            # (rpn_outs[1] has shape: (batch_size, 200, 176, 2)) (2 = 1*num_anchor_per_loc) (conf_score) (just one class (Car))
            # (rpn_outs[2] has shape: (batch_size, 200, 176, 4)) (4 = 2*num_anchor_per_loc) (classification of heading directon (forward or backward))
            # print (len(rpn_outs))
            # print (rpn_outs[0].size())
            # print (rpn_outs[1].size())
            # print (rpn_outs[2].size())

            if not self.full_cfg.SA_SSD_fixed:
                rpn_loss_inputs = rpn_outs + (ret['gt_bboxes'], ret['gt_labels'], ret['anchors'], ret['anchors_mask'], self.train_cfg.rpn)
                rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
                losses.update(rpn_losses)

            guided_anchors = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'], ret['gt_bboxes'], thr=0.1)
            # (guided_anchors is a list of batch_size tensors)
            # (guided_anchors[i] has shape: (num_guided_anchors_in_pc_i, 7))
            # # (num_guided_anchors_in_pc_i is different for different i:s and for different examples) (typically, num_guided_anchors_in_pc_i is ~ 10000 - 25000)
            # # (these are the predicted bboxes (with residuals added to the anchors) with conf_score > 0.1?)
            # print (len(guided_anchors))
            # print (guided_anchors[0].size())
            # print (guided_anchors[1].size())
        else:
            raise NotImplementedError

        # bbox head forward and loss
        if self.extra_head:
            bbox_score = self.extra_head(conv6, guided_anchors)
            # print (bbox_score.size())
            # (bbox_score has shape: (num_guided_anchors_in_batch))
            # # (num_guided_anchors_in_batch = num_guided_anchors_in_pc_0 + num_guided_anchors_in_pc_1 + ... + num_guided_anchors_in_pc_{batch_size - 1})

            if not self.full_cfg.SA_SSD_fixed:
                refine_loss_inputs = (bbox_score, ret['gt_bboxes'], ret['gt_labels'], guided_anchors, self.train_cfg.extra)
                refine_losses = self.extra_head.loss(*refine_loss_inputs)
                losses.update(refine_losses)

        if self.full_cfg is not None:
            if self.full_cfg.USE_EBM:
                # print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                # (conv6 has shape: (batch_size, 256, 200, 176))

                # (ret["gt_bboxes"] is a list of batch_size tensors)
                # (ret["gt_bboxes"][i] has shape: (num_gt_bboxes_i, 7)) (num_gt_bboxes_i can be different for different i:s and for different batches)

                print (conv6.size())
                print (len(ret['gt_bboxes']))
                print (ret['gt_bboxes'][0].size())
                print (ret['gt_bboxes'][1].size())

                batch_size = len(ret['gt_bboxes'])

                ys_list = ret["gt_bboxes"]
                y_samples_list = []
                q_y_samples_list = []
                q_ys_list = []
                for i in range(batch_size):
                    # (ys_list[i] has shape: (num_gt_bboxes_i, 7))
                    # print (ys_list[i].size())

                    y_samples_zero, q_y_samples, q_ys = sample_gmm_centered(self.stds, num_samples=self.num_samples)
                    y_samples_zero = y_samples_zero.cuda() # (shape: (num_samples, 7))
                    q_y_samples = q_y_samples.cuda() # (shape: (num_samples))
                    y_samples = ys_list[i].unsqueeze(1) + y_samples_zero.unsqueeze(0) # (shape: (num_gt_bboxes_i, num_samples, 7))
                    y_samples[:, :, 3:6] = min_hwl + F.relu(y_samples[:, :, 3:6] - min_hwl)
                    y_samples[:, :, 6] = wrapToPi(y_samples[:, :, 6])
                    q_y_samples = q_y_samples.unsqueeze(0)*torch.ones(y_samples.size(0), y_samples.size(1)).cuda() # (shape: (num_gt_bboxes_i, num_samples))
                    q_ys = q_ys[0]*torch.ones(ys_list[i].size(0)).cuda() # (shape: (num_gt_bboxes_i))

                    # print (ys_list[i][0])
                    # print (y_samples_list[i][0, 0:5])

                    y_samples_list.append(y_samples)
                    q_y_samples_list.append(q_y_samples)
                    q_ys_list.append(q_ys)

                    # print (y_samples_list[i].size())
                    # print (q_y_samples_list[i].size())
                    # print (q_ys_list[i].size())
                    # print ("%%%%%")

                ys_features_list = []
                y_samples_features_list = []
                q_y_samples_list_long = []
                for i in range(batch_size):
                    # (conv6 has shape: (batch_size, 256, 200, 176))
                    # (ys_list[i] has shape: (num_gt_bboxes_i, 7))
                    # (y_samples_list[i] has shape: (num_gt_bboxes_i, num_samples, 7))
                    # print (conv6.size())
                    # print (ys_list[i].size())
                    # print (y_samples_list[i].size())

                    (ys_pixel_xs, ys_pixel_ys) = self.gen_grid_fn(ys_list[i][:, [0, 1, 3, 4, 6]])
                    # (both have shape: (4, 7, num_gt_bboxes_i))
                    # print (ys_pixel_xs.size())
                    # print (ys_pixel_ys.size())
                    ys_pixel_xs = ys_pixel_xs.permute(2, 0, 1).contiguous() # (shape: (num_gt_bboxes_i, 4, 7))
                    ys_pixel_ys = ys_pixel_ys.permute(2, 0, 1).contiguous() # (shape: (num_gt_bboxes_i, 4, 7))
                    # print (ys_pixel_xs.size())
                    # print (ys_pixel_ys.size())
                    ys_pixel_coords = torch.cat([ys_pixel_xs.unsqueeze(3), ys_pixel_ys.unsqueeze(3)], 3)
                    # (shape: (num_gt_bboxes_i, 4, 7, 2))
                    # print (ys_pixel_coords.size())

                    conv6_i = conv6[i].unsqueeze(0) # (shape: (1, 256, 200, 176))
                    # print (conv6_i.size())
                    conv6_i_ys = conv6_i.expand(ys_pixel_coords.size(0), -1, -1, -1)
                    # (shape: (num_gt_bboxes_i, 256, 200, 176))
                    # print (conv6_i_ys.size())
                    ys_feature_maps = bilinear_interpolate_torch_gridsample(conv6_i_ys, ys_pixel_coords)
                    # (shape: (num_gt_bboxes_i, 256, 4, 7))
                    # print (ys_feature_maps.size())
                    ys_features = ys_feature_maps.view(ys_feature_maps.size(0), -1)
                    # (shape: (num_gt_bboxes_i, 7168)) (7168 = 256*4*7)
                    # print (ys_features.size())
                    z_feature = F.relu(self.z_fc1(ys_list[i][:, 2].unsqueeze(1))) # (shape: (num_gt_bboxes_i, 16))
                    z_feature = F.relu(self.z_fc2(z_feature)) # (shape: (num_gt_bboxes_i, 16))
                    # print (z_feature.size())
                    h_feature = F.relu(self.h_fc1(ys_list[i][:, 5].unsqueeze(1))) # (shape: (num_gt_bboxes_i, 16))
                    h_feature = F.relu(self.h_fc2(h_feature)) # (shape: (num_gt_bboxes_i, 16))
                    # print (h_feature.size())
                    ys_features = torch.cat([ys_features, z_feature, h_feature], 1) # (shape: (num_gt_bboxes_i, 7168+16+16))
                    # print (ys_features.size())
                    ys_features_list.append(ys_features)

                    for k in range(y_samples_list[i].size(0)):
                        (y_samples_pixel_xs, y_samples_pixel_ys) = self.gen_grid_fn(y_samples_list[i][k, :, [0, 1, 3, 4, 6]])
                        # (both have shape: (4, 7, num_samples))
                        # print (y_samples_pixel_xs.size())
                        # print (y_samples_pixel_ys.size())
                        y_samples_pixel_xs = y_samples_pixel_xs.permute(2, 0, 1).contiguous() # (shape: (num_samples, 4, 7))
                        y_samples_pixel_ys = y_samples_pixel_ys.permute(2, 0, 1).contiguous() # (shape: (num_samples, 4, 7))
                        # print (y_samples_pixel_xs.size())
                        # print (y_samples_pixel_ys.size())
                        y_samples_pixel_coords = torch.cat([y_samples_pixel_xs.unsqueeze(3), y_samples_pixel_ys.unsqueeze(3)], 3)
                        # (shape: (num_samples, 4, 7, 2))
                        # print (y_samples_pixel_coords.size())

                        conv6_i_y_samples = conv6_i.expand(y_samples_pixel_coords.size(0), -1, -1, -1)
                        # (shape: (num_samples, 256, 200, 176))
                        # print (conv6_i_y_samples.size())
                        y_samples_feature_maps = bilinear_interpolate_torch_gridsample(conv6_i_y_samples, y_samples_pixel_coords)
                        # (shape: (num_samples, 256, 4, 7))
                        # print (y_samples_feature_maps.size())
                        y_samples_features = y_samples_feature_maps.view(y_samples_feature_maps.size(0), -1)
                        # (shape: (num_samples, 7168)) (7168 = 256*4*7)
                        # print (y_samples_features.size())
                        z_feature = F.relu(self.z_fc1(y_samples_list[i][k, :, 2].unsqueeze(1))) # (shape: (num_samples, 16))
                        z_feature = F.relu(self.z_fc2(z_feature)) # (shape: (num_samples, 16))
                        # print (z_feature.size())
                        h_feature = F.relu(self.h_fc1(y_samples_list[i][k, :, 5].unsqueeze(1))) # (shape: (num_samples, 16))
                        h_feature = F.relu(self.h_fc2(h_feature)) # (shape: (num_samples, 16))
                        # print (h_feature.size())
                        y_samples_features = torch.cat([y_samples_features, z_feature, h_feature], 1) # (shape: (num_samples, 7168+16+16))
                        # print (y_samples_features.size())
                        y_samples_features_list.append(y_samples_features)

                        q_y_samples_list_long.append(q_y_samples_list[i][k])

                # print (ys_features_list[0].size())
                # print (ys_features_list[1].size())
                ys_features = torch.cat(ys_features_list, 0)
                # (shape: (num_gt_bboxes_in_batch, 7168+16+16))
                print (ys_features.size())

                y_samples_features = torch.cat(y_samples_features_list, 0)
                # (shape: (num_gt_bboxes_in_batch*num_samples, 7168+16+16))
                print (y_samples_features.size())

                features = torch.cat([ys_features, y_samples_features], 0)
                # (shape: (num_gt_bboxes_in_batch + num_gt_bboxes_in_batch*num_samples, 7168+16+16))
                # print (features.size())

                features = F.relu(self.ebm_fc1(features)) # (shape: (num_gt_bboxes_in_batch + num_gt_bboxes_in_batch*num_samples, 1024))
                # print (features.size())
                features = F.relu(self.ebm_fc2(features)) # (shape: (num_gt_bboxes_in_batch + num_gt_bboxes_in_batch*num_samples, 1024))
                # print (features.size())

                fs = self.ebm_fc3(features) # (shape: (num_gt_bboxes_in_batch + num_gt_bboxes_in_batch*num_samples, 1))
                # print (fs.size())
                fs = fs.squeeze(1) # (shape: (num_gt_bboxes_in_batch + num_gt_bboxes_in_batch*num_samples))
                # print (fs.size())

                ys_fs = fs[0:ys_features.size(0)]
                # (shape: (num_gt_bboxes_in_batch))
                # print (ys_fs.size())

                y_samples_fs = fs[ys_features.size(0):]
                # (shape: (num_gt_bboxes_in_batch*num_samples))
                # print (y_samples_fs.size())

                y_samples_fs = y_samples_fs.view(-1, self.num_samples)
                # (shape: (num_gt_bboxes_in_batch, num_samples))
                # print (y_samples_fs.size())

                q_ys = torch.cat(q_ys_list, 0)
                # (shape: (num_gt_bboxes_in_batch))
                # print (q_ys.size())

                q_y_samples = torch.cat(q_y_samples_list_long, 0)
                # (shape: (num_gt_bboxes_in_batch*num_samples))
                # print (q_y_samples.size())

                q_y_samples = q_y_samples.view(-1, self.num_samples)
                # (shape: (num_gt_bboxes_in_batch, num_samples))
                # print (q_y_samples.size())

                # print ("//////////////////")
                # (ys_fs has shape: (num_gt_bboxes_in_batch))
                # (y_samples_fs has shape: (num_gt_bboxes_in_batch, num_samples))
                # (q_ys has shape: (num_gt_bboxes_in_batch))
                # (q_y_samples has shape: (num_gt_bboxes_in_batch, num_samples))

                print (ys_fs.size())
                print (y_samples_fs.size())
                print (q_ys.size())
                print (q_y_samples.size())

                # print (ys_fs[0])
                # print (y_samples_fs[0])
                # print (q_ys)
                # print (q_y_samples[0])

                print ("ys_fs - mean: %f, max: %f, min: %f" % (torch.mean(ys_fs).item(), torch.max(ys_fs).item(), torch.min(ys_fs).item()))
                print ("y_samples_fs - mean: %f, max: %f, min: %f" % (torch.mean(y_samples_fs).item(), torch.max(y_samples_fs).item(), torch.min(y_samples_fs).item()))

                f_samples = y_samples_fs # (shape: (num_gt_bboxes_in_batch, num_samples))
                p_N_samples = q_y_samples # (shape: (num_gt_bboxes_in_batch, num_samples))
                f_0 = ys_fs # (shape: (num_gt_bboxes_in_batch))
                p_N_0 = q_ys # (shape: (num_gt_bboxes_in_batch))
                exp_vals_0 = f_0-torch.log(p_N_0 + 0.0) # (shape: (num_gt_bboxes_in_batch))
                exp_vals_samples = f_samples-torch.log(p_N_samples + 0.0) # (shape: (num_gt_bboxes_in_batch, num_samples))
                exp_vals = torch.cat([exp_vals_0.unsqueeze(1), exp_vals_samples], dim=1) # (shape: (num_gt_bboxes_in_batch, 1+num_samples))
                ebm_loss = -torch.mean(exp_vals_0 - torch.logsumexp(exp_vals, dim=1))
                losses.update(dict(loss_ebm=ebm_loss,))

        print ("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")

        return losses

    def forward_test(self, img, img_meta, **kwargs):
        with torch.no_grad():
            batch_size = len(img_meta) # (batch_size = 1)
            # print (batch_size)

            ret = self.merge_second_batch(kwargs)
            # (ret["voxels"] has shape: (num_voxels, 4)) (num_voxels is different for different examples) (for batch_size = 2, num_voxels is typically 35000 - 45000)
            # (ret["coordinates"] has shape: (num_voxels, 4))
            # print (ret["voxels"].size())
            # print (ret["coordinates"].size())

            vx = self.backbone(ret['voxels'], ret['num_points'])
            # (vx has shape: (num_voxels, 4)) (vx is just identical to ret["voxels"]? seems so)
            # print (vx.size())

            (x, conv6) = self.neck(vx, ret['coordinates'], batch_size, is_test=True)
            # (x has shape: (batch_size, 256, 200, 176))
            # (conv6 has shape: (batch_size, 256, 200, 176))
            # print (x.size())
            # print (conv6.size())

            rpn_outs = self.rpn_head.forward(x)
            # (rpn_outs is a list of 3 elements)
            # (rpn_outs[0] has shape: (batch_size, 200, 176, 14)) (14 = 7*num_anchor_per_loc) (x, y, z, h, w, l, theta)
            # (rpn_outs[1] has shape: (batch_size, 200, 176, 2)) (2 = 1*num_anchor_per_loc) (conf_score) (just one class (Car))
            # (rpn_outs[2] has shape: (batch_size, 200, 176, 4)) (4 = 2*num_anchor_per_loc) (classification of heading directon (forward or backward))
            # print (len(rpn_outs))
            # print (rpn_outs[0].size())
            # print (rpn_outs[1].size())
            # print (rpn_outs[2].size())

            guided_anchors = self.rpn_head.get_guided_anchors(*rpn_outs, ret['anchors'], ret['anchors_mask'], None, thr=.1)
            # (guided_anchors is a list of batch_size tensors)
            # (guided_anchors[i] has shape: (num_guided_anchors_in_pc_i, 7))
            # # (num_guided_anchors_in_pc_i is different for different i:s and for different examples)
            # # (these are the predicted bboxes (with residuals added to the anchors) with conf_score > 0.1?)
            # print (len(guided_anchors))
            # print (guided_anchors[0].size())

            bbox_score, guided_anchors = self.extra_head(conv6, guided_anchors, is_test=True)
            # (bbox_score is a list of batch_size tensors)
            # # (bbox_score[i] has shape: (num_guided_anchors_in_pc_i))
            # (guided_anchors is a list of batch_size tensors)
            # # (guided_anchors[i] has shape: (num_guided_anchors_in_pc_i, 7))
            print (" ")
            print (len(bbox_score))
            print (bbox_score[0].size())
            # print (bbox_score[0])
            print (len(guided_anchors))
            print (guided_anchors[0].size())

            # if self.full_cfg is not None:
            #     if self.full_cfg.USE_EBM and (self.test_cfg.extra.EBM_guided or self.test_cfg.extra.EBM_refine):
            #         # print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            #         # (conv6 has shape: (batch_size, 256, 200, 176))
            #         # (guided_anchors is a list of batch_size tensors)
            #         # (guided_anchors[i] has shape: (num_guided_anchors_in_pc_i, 7))
            #
            #         batch_size = len(guided_anchors)
            #
            #         ys_list = guided_anchors
            #
            #         ys_features_list = []
            #         for i in range(batch_size):
            #             # (conv6 has shape: (batch_size, 256, 200, 176))
            #             # (ys_list[i] has shape: (num_guided_anchors_in_pc_i, 7))
            #             # print (conv6.size())
            #             # print (ys_list[i].size())
            #
            #             if ys_list[i].size(0) == 0:
            #                 ys_features_list.append(None)
            #                 continue
            #
            #             (ys_pixel_xs, ys_pixel_ys) = self.gen_grid_fn(ys_list[i][:, [0, 1, 3, 4, 6]])
            #             # (both have shape: (4, 7, num_guided_anchors_in_pc_i))
            #             # print (ys_pixel_xs.size())
            #             # print (ys_pixel_ys.size())
            #             ys_pixel_xs = ys_pixel_xs.permute(2, 0, 1).contiguous() # (shape: (num_guided_anchors_in_pc_i, 4, 7))
            #             ys_pixel_ys = ys_pixel_ys.permute(2, 0, 1).contiguous() # (shape: (num_guided_anchors_in_pc_i, 4, 7))
            #             # print (ys_pixel_xs.size())
            #             # print (ys_pixel_ys.size())
            #             ys_pixel_coords = torch.cat([ys_pixel_xs.unsqueeze(3), ys_pixel_ys.unsqueeze(3)], 3)
            #             # (shape: (num_guided_anchors_in_pc_i, 4, 7, 2))
            #             # print (ys_pixel_coords.size())
            #
            #             conv6_i = conv6[i].unsqueeze(0) # (shape: (1, 256, 200, 176))
            #             # print (conv6_i.size())
            #             conv6_i_ys = conv6_i.expand(ys_pixel_coords.size(0), -1, -1, -1)
            #             # (shape: (num_guided_anchors_in_pc_i, 256, 200, 176))
            #             # print (conv6_i_ys.size())
            #
            #             if conv6_i_ys.size(0) < 150:
            #                 ys_feature_maps = bilinear_interpolate_torch_gridsample(conv6_i_ys, ys_pixel_coords)
            #                 # (shape: (num_guided_anchors_in_pc_i, 256, 4, 7))
            #                 # print (ys_feature_maps.size())
            #             else:
            #                 num_iters = int(math.floor(conv6_i_ys.size(0)/150.0))
            #                 ys_feature_maps_list = []
            #                 for iter in range(num_iters):
            #                     ys_feature_maps_list.append(bilinear_interpolate_torch_gridsample(conv6_i_ys[(150*iter):(150*(iter+1))], ys_pixel_coords[(150*iter):(150*(iter+1))]))
            #                 ys_feature_maps_list.append(bilinear_interpolate_torch_gridsample(conv6_i_ys[(150*num_iters):], ys_pixel_coords[(150*num_iters):]))
            #                 ys_feature_maps = torch.cat(ys_feature_maps_list, 0)
            #                 # (shape: (num_guided_anchors_in_pc_i, 256, 4, 7))
            #
            #             ys_features = ys_feature_maps.view(ys_feature_maps.size(0), -1)
            #             # (shape: (num_guided_anchors_in_pc_i, 7168)) (7168 = 256*4*7)
            #             # print (ys_features.size())
            #             z_feature = F.relu(self.z_fc1(ys_list[i][:, 2].unsqueeze(1))) # (shape: (num_guided_anchors_in_pc_i, 16))
            #             z_feature = F.relu(self.z_fc2(z_feature)) # (shape: (num_guided_anchors_in_pc_i, 16))
            #             # print (z_feature.size())
            #             h_feature = F.relu(self.h_fc1(ys_list[i][:, 5].unsqueeze(1))) # (shape: (num_guided_anchors_in_pc_i, 16))
            #             h_feature = F.relu(self.h_fc2(h_feature)) # (shape: (num_guided_anchors_in_pc_i, 16))
            #             # print (h_feature.size())
            #             ys_features = torch.cat([ys_features, z_feature, h_feature], 1) # (shape: (num_guided_anchors_in_pc_i, 7168+16+16))
            #             # print (ys_features.size())
            #             ys_features_list.append(ys_features)
            #
            #         fs_list = []
            #         for i in range(batch_size):
            #             if ys_features_list[i] is None:
            #                 fs_list.append(None)
            #                 continue
            #
            #             features = F.relu(self.ebm_fc1(ys_features_list[i])) # (shape: (num_guided_anchors_in_pc_i, 1024))
            #             # print (features.size())
            #             features = F.relu(self.ebm_fc2(features)) # (shape: (num_guided_anchors_in_pc_i, 1024))
            #             # print (features.size())
            #
            #             fs = self.ebm_fc3(features) # (shape: (num_guided_anchors_in_pc_i, 1))
            #             # print (fs.size())
            #             fs = fs.squeeze(1) # (shape: (num_guided_anchors_in_pc_i))
            #             # print (fs.size())
            #
            #             fs_list.append(fs)
            #
            #     # (fs_list is a list of batch_size tensors)
            #     # # (fs_list[i] has shape: (num_guided_anchors_in_pc_i))
            #     # print (" ")
            #     # print (len(fs_list))
            #     # print (fs_list[0].size())
            #     # print (fs_list[0])

            if self.test_cfg.extra.EBM_guided:
                det_bboxes, det_scores, det_fs = self.extra_head.get_rescore_bboxes_ebm_guided(
                    guided_anchors, bbox_score, fs_list, img_meta, self.test_cfg.extra)
            else:
                det_bboxes, det_scores = self.extra_head.get_rescore_bboxes(
                    guided_anchors, bbox_score, img_meta, self.test_cfg.extra)
                det_fs = None
            # (det_scores is a list of batch_size numpy arrays)
            # # (det_scores[i] has shape: (num_detections_i)) (num_detections_i <= num_guided_anchors_in_pc_i)
            # (det_fs is a list of batch_size numpy arrays)
            # # (det_fs[i] has shape: (num_detections_i))
            # (det_bboxes is a list of batch_size numpy arrays)
            # # (det_bboxes[i] has shape: (num_detections_i, 7))
            # print (len(det_scores))
            # print (det_scores[0].shape)
            # print (len(det_bboxes))
            # print (det_bboxes[0].shape)

            print (" ")
            print ("fs before refinement:")
            if det_fs is not None:
                print (det_fs.detach().cpu().numpy())
            else:
                print (det_fs)
            print ("bboxes before refinement:")
            print (det_bboxes[0])
            # print ("%%%%%%%%%%%%%%%%%%%%%%")
        # (end of "with torch.no_grad():"") ####################################

        if self.test_cfg.extra.EBM_refine:
            # (det_bboxes is a list of batch_size numpy arrays)
            # # (det_bboxes[i] has shape: (num_detections_i, 7))
            # (conv6 has shape: (batch_size, 256, 200, 176))
            # (batch_size == 1)
            # print (conv6.size())

            bboxes = []
            for i in range(len(det_bboxes)):
                bboxes.append(torch.from_numpy(det_bboxes[i]).cuda())
            # (bboxes is a list of batch_size tensors)
            # # (bboxes[i] has shape: (num_detections_i, 7))
            # print (len(bboxes))
            # print (bboxes[0].size())

            conv6.requires_grad = True

            det_bboxes = []
            for i in range(len(bboxes)):
                # (conv6 has shape: (batch_size, 256, 200, 176))
                # (bboxes[i] has shape: (num_detections_i, 7))
                # print (conv6.size())
                # print (bboxes[i].size())

                bboxes_i = bboxes[i] # (shape: (num_detections_i, 7))
                if bboxes_i.size(0) == 0:
                    det_bboxes.append(bboxes_i.detach().cpu().numpy())
                    continue

                step_sizes = 0.0002*torch.ones(bboxes_i.size(0), 1).cuda() # (shape: (num_detections_i, 1))
                print (self.test_cfg.extra.EBM_refine_steps)
                print (step_sizes)
                for step in range(self.test_cfg.extra.EBM_refine_steps):
                    # print (step_sizes)

                    bboxes_init = bboxes_i.clone().detach() # (shape: (num_detections_i, 7))

                    bboxes_init.requires_grad = True

                    # print (bboxes_init[0])
                    # print (bboxes_init.size())

                    ############################################################
                    ############################################################
                    ############################################################
                    ############################################################
                    ############################################################
                    (ys_pixel_xs, ys_pixel_ys) = self.gen_grid_fn(bboxes_init[:, [0, 1, 3, 4, 6]])
                    # (both have shape: (4, 7, num_detections_i))
                    # print (ys_pixel_xs.size())
                    # print (ys_pixel_ys.size())
                    ys_pixel_xs = ys_pixel_xs.permute(2, 0, 1).contiguous() # (shape: (num_detections_i, 4, 7))
                    ys_pixel_ys = ys_pixel_ys.permute(2, 0, 1).contiguous() # (shape: (num_detections_i, 4, 7))
                    # print (ys_pixel_xs.size())
                    # print (ys_pixel_ys.size())
                    ys_pixel_coords = torch.cat([ys_pixel_xs.unsqueeze(3), ys_pixel_ys.unsqueeze(3)], 3)
                    # (shape: (num_detections_i, 4, 7, 2))
                    # print (ys_pixel_coords.size())
                    #
                    conv6_i = conv6[i].unsqueeze(0) # (shape: (1, 256, 200, 176))
                    # print (conv6_i.size())
                    conv6_i_ys = conv6_i.expand(ys_pixel_coords.size(0), -1, -1, -1)
                    # (shape: (num_detections_i, 256, 200, 176))
                    # print (conv6_i_ys.size())
                    #
                    if conv6_i_ys.size(0) < 150:
                        ys_feature_maps = bilinear_interpolate_torch_gridsample(conv6_i_ys, ys_pixel_coords)
                        # (shape: (num_detections_i, 256, 4, 7))
                        # print (ys_feature_maps.size())
                    else:
                        num_iters = int(math.floor(conv6_i_ys.size(0)/150.0))
                        ys_feature_maps_list = []
                        for iter in range(num_iters):
                            ys_feature_maps_list.append(bilinear_interpolate_torch_gridsample(conv6_i_ys[(150*iter):(150*(iter+1))], ys_pixel_coords[(150*iter):(150*(iter+1))]))
                        ys_feature_maps_list.append(bilinear_interpolate_torch_gridsample(conv6_i_ys[(150*num_iters):], ys_pixel_coords[(150*num_iters):]))
                        ys_feature_maps = torch.cat(ys_feature_maps_list, 0)
                        # (shape: (num_detections_i, 256, 4, 7))
                    #
                    ys_features = ys_feature_maps.view(ys_feature_maps.size(0), -1)
                    # (shape: (num_detections_i, 7168)) (7168 = 256*4*7)
                    # print (ys_features.size())
                    z_feature = F.relu(self.z_fc1(bboxes_init[:, 2].unsqueeze(1))) # (shape: (num_detections_i, 16))
                    z_feature = F.relu(self.z_fc2(z_feature)) # (shape: (num_detections_i, 16))
                    # print (z_feature.size())
                    h_feature = F.relu(self.h_fc1(bboxes_init[:, 5].unsqueeze(1))) # (shape: (num_detections_i, 16))
                    h_feature = F.relu(self.h_fc2(h_feature)) # (shape: (num_detections_i, 16))
                    # print (h_feature.size())
                    ys_features = torch.cat([ys_features, z_feature, h_feature], 1) # (shape: (num_detections_i, 7168+16+16))
                    # print (ys_features.size())
                    #
                    features = F.relu(self.ebm_fc1(ys_features)) # (shape: (num_detections_i, 1024))
                    # print (features.size())
                    features = F.relu(self.ebm_fc2(features)) # (shape: (num_detections_i, 1024))
                    # print (features.size())
                    #
                    fs = self.ebm_fc3(features) # (shape: (num_detections_i, 1))
                    # print (fs.size())
                    fs = fs.squeeze(1) # (shape: (num_detections_i))
                    # print (fs.size())
                    # print (fs)
                    ############################################################
                    ############################################################
                    ############################################################
                    ############################################################
                    ############################################################

                    grad_bboxes_init = torch.autograd.grad(fs.sum(), bboxes_init, create_graph=True)[0]
                    # (shape: (num_detections_i, 7)) (same as bboxes_init)
                    # print (grad_bboxes_init.size())

                    bboxes_refined = bboxes_init + step_sizes*grad_bboxes_init

                    with torch.no_grad():
                        ############################################################
                        ############################################################
                        ############################################################
                        ############################################################
                        ############################################################
                        (ys_pixel_xs, ys_pixel_ys) = self.gen_grid_fn(bboxes_refined[:, [0, 1, 3, 4, 6]])
                        # (both have shape: (4, 7, num_detections_i))
                        # print (ys_pixel_xs.size())
                        # print (ys_pixel_ys.size())
                        ys_pixel_xs = ys_pixel_xs.permute(2, 0, 1).contiguous() # (shape: (num_detections_i, 4, 7))
                        ys_pixel_ys = ys_pixel_ys.permute(2, 0, 1).contiguous() # (shape: (num_detections_i, 4, 7))
                        # print (ys_pixel_xs.size())
                        # print (ys_pixel_ys.size())
                        ys_pixel_coords = torch.cat([ys_pixel_xs.unsqueeze(3), ys_pixel_ys.unsqueeze(3)], 3)
                        # (shape: (num_detections_i, 4, 7, 2))
                        # print (ys_pixel_coords.size())
                        #
                        conv6_i = conv6[i].unsqueeze(0) # (shape: (1, 256, 200, 176))
                        # print (conv6_i.size())
                        conv6_i_ys = conv6_i.expand(ys_pixel_coords.size(0), -1, -1, -1)
                        # (shape: (num_detections_i, 256, 200, 176))
                        # print (conv6_i_ys.size())
                        #
                        if conv6_i_ys.size(0) < 150:
                            ys_feature_maps = bilinear_interpolate_torch_gridsample(conv6_i_ys, ys_pixel_coords)
                            # (shape: (num_detections_i, 256, 4, 7))
                            # print (ys_feature_maps.size())
                        else:
                            num_iters = int(math.floor(conv6_i_ys.size(0)/150.0))
                            ys_feature_maps_list = []
                            for iter in range(num_iters):
                                ys_feature_maps_list.append(bilinear_interpolate_torch_gridsample(conv6_i_ys[(150*iter):(150*(iter+1))], ys_pixel_coords[(150*iter):(150*(iter+1))]))
                            ys_feature_maps_list.append(bilinear_interpolate_torch_gridsample(conv6_i_ys[(150*num_iters):], ys_pixel_coords[(150*num_iters):]))
                            ys_feature_maps = torch.cat(ys_feature_maps_list, 0)
                            # (shape: (num_detections_i, 256, 4, 7))
                        #
                        ys_features = ys_feature_maps.view(ys_feature_maps.size(0), -1)
                        # (shape: (num_detections_i, 7168)) (7168 = 256*4*7)
                        # print (ys_features.size())
                        z_feature = F.relu(self.z_fc1(bboxes_refined[:, 2].unsqueeze(1))) # (shape: (num_detections_i, 16))
                        z_feature = F.relu(self.z_fc2(z_feature)) # (shape: (num_detections_i, 16))
                        # print (z_feature.size())
                        h_feature = F.relu(self.h_fc1(bboxes_refined[:, 5].unsqueeze(1))) # (shape: (num_detections_i, 16))
                        h_feature = F.relu(self.h_fc2(h_feature)) # (shape: (num_detections_i, 16))
                        # print (h_feature.size())
                        ys_features = torch.cat([ys_features, z_feature, h_feature], 1) # (shape: (num_detections_i, 7168+16+16))
                        # print (ys_features.size())
                        #
                        features = F.relu(self.ebm_fc1(ys_features)) # (shape: (num_detections_i, 1024))
                        # print (features.size())
                        features = F.relu(self.ebm_fc2(features)) # (shape: (num_detections_i, 1024))
                        # print (features.size())
                        #
                        new_fs = self.ebm_fc3(features) # (shape: (num_detections_i, 1))
                        # print (new_fs.size())
                        new_fs = new_fs.squeeze(1) # (shape: (num_detections_i))
                        # print (new_fs.size())
                        # print (new_fs)
                        ############################################################
                        ############################################################
                        ############################################################
                        ############################################################
                        ############################################################

                        refinement_failed = (new_fs < fs) # (shape: (num_detections_i))
                        # print (refinement_failed)
                        # print (refinement_failed.size())
                        refinement_failed = refinement_failed.unsqueeze(1) # (shape: (num_detections_i, 1))
                        r_f = refinement_failed.float()

                        bboxes_i = r_f*bboxes_init + (1.0-r_f)*bboxes_refined

                        step_sizes = (1.0-r_f)*step_sizes + r_f*0.5*step_sizes

                        if step == self.test_cfg.extra.EBM_refine_steps - 1: # (in final step)
                            refinement_failed = (new_fs < fs) # (shape: (num_detections_i))
                            # print (refinement_failed)
                            # print (refinement_failed.size())
                            r_f = refinement_failed.float()
                            final_fs = r_f*fs + (1.0-r_f)*new_fs

                            print ("###")
                            print ("###")
                            print ("###")
                            print ("fs after refinement:")
                            print (final_fs.detach().cpu().numpy())

                det_bboxes.append(bboxes_i.detach().cpu().numpy())

        print ("bboxes after refinement:")
        print (det_bboxes[0])

        #################################################################################
        # uncomment these lines to save the predictions when running on a test sequence:
        #################################################################################
        # sample_idx = img_meta[0]["sample_idx"]
        # import pickle
        # # with open("preds0011/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0002/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0007/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0001/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0000/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0003/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0004/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0005/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0006/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0008/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0009/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0010/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0012/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0013/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0014/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0015/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0016/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0017/%d.pkl" % sample_idx, "wb") as file:
        # # with open("preds0018/%d.pkl" % sample_idx, "wb") as file:
        # with open("preds0027/%d.pkl" % sample_idx, "wb") as file:
        #     pickle.dump(det_bboxes[0], file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
        ########################################################################

        results = [kitti_bbox2results(*param) for param in zip(det_bboxes, det_scores, img_meta)]

        print ("{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}")

        return results
