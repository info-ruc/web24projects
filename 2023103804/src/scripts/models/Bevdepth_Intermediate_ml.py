"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler

Intermediate fusion for camera based collaboration
"""

from numpy import record
import torch
from sklearn.neighbors import KNeighborsRegressor
from torch import nn
import torch.nn.functional as F
from opencood.models.lift_splat_shoot import LiftSplatShoot
from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum, bin_depths
from opencood.models.sub_modules.lss_submodule import BevEncodeMSFusion, Up, CamEncode, CamEncodeGTDepth, BevEncode
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from matplotlib import pyplot as plt
from opencood.models.sub_modules.depthnet import DepthNet, DepthAggregation
from mmdet3d.models.backbones import ResNet
from mmdet3d.models.necks import SECONDFPN
import seaborn as sns


def FusionNet(fusion_args):
    if fusion_args['core_method'] == "max":
        from opencood.models.fuse_modules.max_fuse import MaxFusion
        return MaxFusion(fusion_args['args'])
    elif fusion_args['core_method'] == 'att':
        from opencood.models.fuse_modules.att_fuse import AttFusion
        return AttFusion(fusion_args['args'])
    elif fusion_args['core_method'] == 'v2vnet':
        from opencood.models.fuse_modules.v2v_fuse import V2VNetFusion
        return V2VNetFusion(fusion_args['args'])
    elif fusion_args['core_method'] == 'v2xvit':
        from opencood.models.fuse_modules.v2xvit_fuse import V2XViTFusion
        return V2XViTFusion(fusion_args['args'])
    elif fusion_args['core_method'] == 'when2comm':
        from opencood.models.fuse_modules.when2com_fuse import When2comFusion
        return When2comFusion(fusion_args['args'])
    else:
        raise("Fusion method not implemented.")

class BevdepthIntermediateml(LiftSplatShoot):
    def __init__(self, args):
        super(BevdepthIntermediateml, self).__init__(args)

        fusion_args = args['fusion_args']
        self.ms = args['fusion_args']['core_method'].endswith("ms")
        if self.ms:
            self.bevencode = BevEncodeMSFusion(fusion_args)
        else:
            self.fusion = FusionNet(fusion_args)
        self.supervise_single = args['supervise_single']
        self.d_min = self.grid_conf['ddiscr'][0]
        self.d_max = self.grid_conf['ddiscr'][1]
        self.num_bins = self.grid_conf['ddiscr'][2]
        self.mode = self.grid_conf['mode']
        #TODO: context_channels in bevdepth is class num ?
        self.depthnet = DepthNet(512,512,self.outC,self.D)
        self.use_da = True
        self.depth_aggregation_net = DepthAggregation(self.outC,self.outC,self.outC)
        self.imagebackbone = ResNet(
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=0,
            norm_eval=False,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        )
        self.neck = SECONDFPN(
            in_channels=[256, 512, 1024, 2048],
            # upsample_strides=[0.25, 0.5, 1, 2],
            upsample_strides=[0.5,1,2,4],
            out_channels=[128, 128, 128, 128]
        )

        self.cls_head_2d = nn.Sequential(
                                    nn.Conv2d(512, 32,
                                        kernel_size=1, padding=0, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 1,
                                        kernel_size=1, padding=0, bias=True))

        self.max_pool = nn.MaxPool2d(kernel_size=8, stride=8, padding=0)

    def fill_depth(self,depth):
        N, H, W = depth.size()
        depth = depth.view(N,H,W)
        # depth_downsampled = F.interpolate(depth.unsqueeze(1), size=(40, 60), mode='bilinear',
        #                                   align_corners=True).squeeze(1)

        # data = depth_downsampled.to('cpu').numpy()
        # sns.heatmap(data[0], cmap="viridis", cbar=True, vmin=0, vmax=50, xticklabels=80, yticklabels=80)
        # plt.show()
        # plt.close()

        # 获取有深度值的像素的位置和值
        known_depth_coords = torch.nonzero(depth, as_tuple=False)
        known_depth_values = depth[
            known_depth_coords[:, 0], known_depth_coords[:, 1], known_depth_coords[:, 2]]

        # 获取缺失深度值的像素的位置
        unknown_depth_coords = torch.nonzero(depth == 0, as_tuple=False)

        if len(known_depth_coords) > 0 and len(unknown_depth_coords) > 0:
            # 使用KNN插值来估算缺失的深度值
            knn = KNeighborsRegressor(n_neighbors=3)
            knn.fit(known_depth_coords[:, 1:].cpu().numpy(), known_depth_values.cpu().numpy())
            estimated_depth_values = knn.predict(unknown_depth_coords[:, 1:].cpu().numpy())
            depth[
                unknown_depth_coords[:, 0], unknown_depth_coords[:, 1], unknown_depth_coords[:, 2]] = torch.tensor(
                estimated_depth_values, device=depth.device)

        # data = depth_downsampled.to('cpu').numpy()
        # sns.heatmap(data[0], cmap="viridis", cbar=True, vmin=0, vmax=50, xticklabels=80, yticklabels=80)
        # plt.show()
        # plt.close()
        return depth

    def depthforward(self,x,rots,trans,intrins,post_rots,post_trans,transform_matrix,lidar_depth):
        batch_size,  num_cams, num_channels, img_height, \
            img_width = x.shape
        # get feature map: N,4,512,20,30
        img_feats = self.get_cam_feats(x)

        # _,_,C,H,W = img_feats.shape
        # cls_feat = img_feats.reshape(batch_size*num_cams,C,H,W)
        # cls_2d = self.cls_head_2d(cls_feat)
        # cls_2d = cls_2d.sigmoid()
        # lidar_depth = self.fill_depth(lidar_depth,cls_2d)

        # N*4,1,40,60

        # lidar_depth = lidar_depth.reshape(
        #     batch_size * num_cams, *lidar_depth.shape[2:]).squeeze()

        # depth_gt: N*4,48,40,60   depth_gt_indices:N*4,40,60
        # depth_gt, depth_gt_indices = self.get_gt_depth_dist(lidar_depth)

        # depth_feature: N*4,D+256 ,20,30
        depth_feature = self.depthnet(img_feats,rots,trans,intrins,post_rots,post_trans,transform_matrix)
        # need to supervise depth

        depth = depth_feature[:, :self.D].softmax(
            dim=1, dtype=depth_feature.dtype)


        #get frustum
        geom = self.get_geometry(rots, trans, intrins, post_rots,
                                 post_trans)  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3 (4 x N x 42 x 16 x 22 x 3)

        img_feat_with_depth = depth.unsqueeze(
            1) * depth_feature[:, self.D:(
                self.D + self.outC)].unsqueeze(2)

        img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)
        img_feat_with_depth = img_feat_with_depth.reshape(
            batch_size,
            num_cams,
            img_feat_with_depth.shape[1],
            img_feat_with_depth.shape[2],
            img_feat_with_depth.shape[3],
            img_feat_with_depth.shape[4],
        )

        img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)  # N,4,D,H,W,C

        # geom: N,4,48,44,64,3
        x = self.voxel_pooling(geom, img_feat_with_depth)  # x: 4 x 64 x 240 x 240

        return x,depth
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        if len(gt_depths.shape) == 3:
            gt_depths = gt_depths.unsqueeze(0)
        B, N, H, W = gt_depths.shape
        gt_depths  = self.max_pool(gt_depths)
        # gt_depths = gt_depths.view(
        #     B * N,
        #     H // self.downsample,
        #     self.downsample,
        #     W // self.downsample,
        #     self.downsample,
        #     1,
        # )
        # gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        # gt_depths = gt_depths.view(
        #     -1, self.downsample * self.downsample)
        # gt_depths_tmp = torch.where(gt_depths == 0.0,
        #                             1e5 * torch.ones_like(gt_depths),
        #                             gt_depths)
        # gt_depths = torch.min(gt_depths, dim=-1).values
        # gt_depths = gt_depths.view(B * N, H // self.downsample,
        #                            W // self.downsample)

        gt_depths = (gt_depths -
                     (self.grid_conf['ddiscr'][0] - 1)) / 1
        gt_depths = torch.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.D + 1).view(
                                  -1, self.D + 1)[:, 1:]

        return gt_depths.float()

    def get_downsampled_lidar_depth(self, lidar_depth):
        batch_size, num_cams, height, width = lidar_depth.shape
        lidar_depth = lidar_depth.view(
            batch_size * num_cams,
            height // self.downsample,
            self.downsample,
            width // self.downsample,
            self.downsample,
            1,
        )
        lidar_depth = lidar_depth.permute(0, 1, 3, 5, 2, 4).contiguous()
        lidar_depth = lidar_depth.view(
            -1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(lidar_depth == 0.0, lidar_depth.max(),
                                    lidar_depth)
        lidar_depth = torch.min(gt_depths_tmp, dim=-1).values
        lidar_depth = lidar_depth.view(batch_size, num_cams, 1,
                                       height // self.downsample,
                                       width // self.downsample)
        lidar_depth = lidar_depth / self.grid_conf['ddiscr'][1]
        return lidar_depth

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape  # B: 4  N: 4  C: 3  imH: 320  imW: 480

        x = x.view(B*N, C, imH, imW)

        img_feats = self.neck(self.imagebackbone(x))[0]

        # C = 512
        img_feats = img_feats.reshape(B, N,
                                      img_feats.shape[1], img_feats.shape[2],
                                      img_feats.shape[3])

        return img_feats

    def get_gt_depth_dist(self, x):  # 对深度维进行onehot，得到每个像素不同深度的概率
        """
        Args:
            x: [B*N, H, W]
        Returns:
            x: [B*N, D, fH, fW]
        """
        assert len(x.shape) == 3
        target = self.training
        torch.clamp_max_(x, self.d_max) # save memory
        # [B*N, H, W], indices (float), value: [0, num_bins)
        depth_indices, mask = bin_depths(x, self.mode, self.d_min, self.d_max, self.num_bins, target=target)
        # depth_indices = depth_indices[:, self.downsample//2::self.downsample, self.downsample//2::self.downsample]
        onehot_dist = F.one_hot(depth_indices.long()).permute(0,3,1,2) # [B*N, num_bins, fH, fW]

        if not target:
            # mask = mask[:, self.downsample//2::self.downsample, self.downsample//2::self.downsample].unsqueeze(1)
            mask = mask.unsqueeze(1)
            onehot_dist *= mask

        return onehot_dist, depth_indices

    def _forward_voxel_net(self, img_feat_with_depth):
        if self.use_da:
            # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
            img_feat_with_depth = img_feat_with_depth.permute(
                0, 3, 1, 4,
                2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
            n, h, c, w, d = img_feat_with_depth.shape
            img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
            img_feat_with_depth = (
                self.depth_aggregation_net(img_feat_with_depth).view(
                    n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous())
        return img_feat_with_depth


    def forward(self, data_dict):
        if self.ms:
            return self._forward_ms(data_dict)
        else:
            return self._forward_ss(data_dict)

    def weighted_average_downsample(self,depth_maps, factor=8):
        B, H, W = depth_maps.shape  # Batch size, Height, Width
        new_H, new_W = H // factor, W // factor

        # Reshape to include blocks
        depth_maps = depth_maps.view(B, new_H, factor, new_W, factor)

        # Compute weights (inverse of depth, avoiding division by zero)
        weights = torch.where(depth_maps > 0, 1.0 / depth_maps, torch.zeros_like(depth_maps))

        # Compute weighted sum
        weighted_sum = torch.sum(depth_maps * weights, dim=(2, 4))

        # Compute sum of weights
        sum_weights = torch.sum(weights, dim=(2, 4))

        # Compute weighted average, avoiding division by zero
        downsampled_maps = torch.where(sum_weights > 0, weighted_sum / sum_weights, torch.zeros_like(weighted_sum))

        return downsampled_maps

    def get_downsampled_gt_depth_weight(self,gt_depths, D=48):
        if len(gt_depths.shape) == 2:
            gt_depths = gt_depths.unsqueeze(0)

        N, H, W = gt_depths.shape

        # Perform weighted average downsample
        downsampled_maps = self.weighted_average_downsample(gt_depths)

        # Now downsampled_maps is [N, h, w]
        downsampled_maps = (downsampled_maps - (self.grid_conf['ddiscr'][0] - 1)) / 1

        downsampled_maps = torch.where(
            (downsampled_maps < D + 1) & (downsampled_maps >= 0.0),
            downsampled_maps, torch.zeros_like(downsampled_maps))

        downsampled_maps = F.one_hot(downsampled_maps.long(), num_classes=D + 1).view(-1, D + 1)[:, 1:]

        return downsampled_maps.float()




    def _forward_ms(self, data_dict):
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        image_inputs_dict = data_dict['image_inputs']
        record_len = data_dict['record_len']
        lidar_depth = data_dict['depth_group']
        x, rots, trans, intrins, post_rots, post_trans,transform_matrix = \
            (image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'],
             image_inputs_dict['post_rots'], image_inputs_dict['post_trans'], image_inputs_dict['transformation_matrix'])


        # lidar_depth = self.fill_depth(lidar_depth)
        # as gt_depth
        lidar_depth = self.get_downsampled_gt_depth(lidar_depth)

        # lidar_depth = self.get_downsampled_gt_depth_weight(lidar_depth)
        bev_map, depth  = self.depthforward(x,rots,trans,intrins,post_rots,post_trans,transform_matrix,lidar_depth)

        x_single, x_fuse = self.bevencode(bev_map, record_len, pairwise_t_matrix)
        psm = self.cls_head(x_fuse)
        rm = self.reg_head(x_fuse)

        _,D,H,W = depth.shape
        depth = depth.view(record_len,4,D,H,W)
        depth = depth[0].permute(0, 2, 3, 1).contiguous().view(
            -1, self.D)
        depth_items = [lidar_depth,depth]

        # TODO: add gt_depth and pre_depth
        output_dict = {'psm': psm,
                       'rm': rm,
                       'depth_items': depth_items
                       }
        if self.use_dir:
            dm = self.dir_head(x_fuse)
            output_dict.update({"dm": dm})

        if self.supervise_single:
            psm_single = self.cls_head(x_single)
            rm_single = self.reg_head(x_single)
            output_dict.update({'psm_single': psm_single,
                                'rm_single': rm_single})
            if self.use_dir:
                dm_single = self.dir_head(x_single)
                output_dict.update({"dm_single": dm_single})

        return output_dict

    def _forward_ss(self, data_dict):
        # x:[sum(record_len), 4, 3or4, 256, 352]
        # rots: [sum(record_len), 4, 3, 3]
        # trans: [sum(record_len), 4, 3]
        # intrins: [sum(record_len), 4, 3, 3]
        # post_rots: [sum(record_len), 4, 3, 3]
        # post_trans: [sum(record_len), 4, 3]
        image_inputs_dict = data_dict['image_inputs']
        record_len = data_dict['record_len']
        x, rots, trans, intrins, post_rots, post_trans = \
            image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
        x, depth_items = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)  # 将图像转换到BEV下，x: sum(record_len) x C x 240 x 240 (4 x 64 x 240 x 240)

        x = self.bevencode(x)  # 用resnet18提取特征  x: sum(record_len) x C x 240 x 240
        if self.shrink_flag:
            x = self.shrink_conv(x)
        # 4 x C x 120 x 120

        ## fusion ##
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        x_fuse = self.fusion(x, record_len, pairwise_t_matrix)
        ############

        psm = self.cls_head(x_fuse)
        rm = self.reg_head(x_fuse)
        output_dict = {'psm': psm,
                       'rm': rm,
                       'depth_items': depth_items}

        if self.use_dir:
            dm = self.dir_head(x_fuse)
            output_dict.update({"dm": dm})

        if self.supervise_single:
            psm_single = self.cls_head(x)
            rm_single = self.reg_head(x)
            output_dict.update({'psm_single': psm_single,
                                'rm_single': rm_single})
            if self.use_dir:
                dm_single = self.dir_head(x)
                output_dict.update({"dm_single": dm_single})

        return output_dict


def compile_model(grid_conf, data_aug_conf, outC):
    return BevdepthIntermediateml(grid_conf, data_aug_conf, outC)