
import torch
import math
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from opencood.models.sub_modules.lss_submodule import BevEncodeMSFusion, mulBEVfusion
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from matplotlib import pyplot as plt

from opencood.models.spatial_cross_attention import SpatialCrossAttention,  SimpleNetwork
from opencood.models.sub_modules.fpn import FPN
from opencood.models.sub_modules.ref_3d import *
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

from opencood.utils.common_utils import get_true_lidar2img_matrix
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class BevFspatialcos(nn.Module):
    def __init__(self, args):
        super(BevFspatialcos, self).__init__()
        # any para will be put into yaml
        fusion_args = args['fusion_args']
        self.data_aug_conf = args['data_aug_conf']  # 数据增强配置参数
        self.bev_h = args['bev_h']
        self.bev_w = args['bev_w']
        self.outC = args['outC']
        self.embed_dims = args['img_features_C']
        self.downsample = args['img_downsample']  # 下采样倍数
        self.pc_range = args['gt_range']
        self.num_points_in_pillar = args['num_points_in_pillar']
        self.num_cams = args['Ncams']
        self.num_feature_levels = args['num_feature_levels']
        # bev_query
        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims)

        self.use_cams_embeds = args['use_cams_embeds']
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        init.normal_(self.cams_embeds)
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        init.normal_(self.level_embeds)


        self.cycle = args['cycle']
        self.discrete_ratio = fusion_args['args']['voxel_size'][0]
        self.downsample_rate = 1
        # get feature map
        self.get_mul_fea = FPN(self.num_feature_levels)

        # self.bevfusion = BevFusion(inC=self.camC, outC=self.outC)
        self.sca = SpatialCrossAttention().to('cuda')

        self.tsblock = SimpleNetwork(self.embed_dims,self.embed_dims)

        self.bevfusion = mulBEVfusion(fusion_args)

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        self.cls_head = nn.Conv2d(self.outC, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.outC, 7 * args['anchor_number'],
                                  kernel_size=1)

        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.outC, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        else:
            self.use_dir = False


    def forward(self,data_dict):
        image_inputs_dict = data_dict['image_inputs']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        lidar2img_matrix = data_dict['lidar2img_matrix']
        x, rots, trans, intrins, post_rots, post_trans = \
            image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], \
                image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
        # B means N agent ,batch_size = 1 , N means num_camera
        # plt.imshow(x[0][0].permute(1,2,0).cpu().numpy())
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()
        # plt.close()

        B, N, C, H, W = x.shape
        batch_size = 1
        # pre bev_query.weight
        bev_query = self.bev_embedding.weight

        # bev_mask = torch.zeros(bs, bev_h, bev_w)

        # we only need one bev
        bev_query = bev_query.unsqueeze(1).repeat(1, batch_size, 1)

        bev_pos = positionalencoding2d(self.embed_dims, self.bev_h, self.bev_w)
        bev_pos = bev_pos.permute(1, 2, 0)
        bev_pos = bev_pos.view(1, -1, self.embed_dims)
        bev_pos = bev_pos.to('cuda')

        # B * N
        x = x.view(-1,C,H,W)
        mlv_fea = self.get_mul_fea(x)
        feat_flatten = []
        spatial_shapes = []


        for lvl, feat in enumerate(mlv_fea):
            bs, c, h, w = feat.shape
            feat = feat.unsqueeze(0).reshape(-1, self.num_cams, c, h, w)
            spatial_shape = (h, w)
            # 变成 num_cam,bs,h*w,C (6,1,116*200,256)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                # cams_embeds: (num_cas,bs,h*w,C)
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            # level_embeds: 1 × 256
            feat = feat + self.level_embeds[None,
                          None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        # 把所有的feature map 一维化，全部cat到 第三个维度，也就是feature map一维化的维度
        feat_flatten = torch.cat(feat_flatten, 2)
        # 3×2 维度的 feature map大小
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long).to('cuda')
        # 每一个维度的一维化的起始点， 0 ~ 75000 ， 75000~ 94000  ， 最后一个不用，，因为剩下的就是他
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)


        # build ref_3d  ref_3d: 4，4，40000，3  -->
        ref_3d = get_reference_points(
        self.bev_h, self.bev_w, self.pc_range[5] - self.pc_range[2], self.num_points_in_pillar,
            bs=B)

        bev_query = bev_query.permute(1, 0, 2)

        # lidar2img_matrix = lidar2img_matrix.squeeze(0)
        lidar2img_matrix_true = get_true_lidar2img_matrix(x,lidar2img_matrix,record_len)


        ogfH, ogfW = self.data_aug_conf['final_dim']
        reference_points_cam, bev_mask = point_sampling(
            ref_3d, self.pc_range, lidar2img_matrix_true, ogfH, ogfW , rots, trans, intrins, post_rots, post_trans)

        # reference_points_cam, bev_mask = point_sampling_handwrite(
        #     ref_3d, self.pc_range, lidar2img_matrix_true, ogfH, ogfW , rots, trans, intrins, post_rots, post_trans)

        bev_groups = []
        for b in range(B):
            # need to distinguish ego and neighbor   only used in batch_size=1

            # split feat ,  reference cam ,  bev_mask
            current_bev_mask = bev_mask[:,b:b+1,...]
            current_reference_points_cam = reference_points_cam[:,b:b+1,...]
            current_feat_flatten = feat_flatten[:,:,b:b+1,:]

            if b==0:
                for i in range(2):
                    op = self.sca(bev_query, current_feat_flatten, current_feat_flatten,None, bev_pos, spatial_shapes, current_reference_points_cam, current_bev_mask,
                     level_start_index)
                    bev_query = self.tsblock(op)

                ego_bev = bev_query
                ego_bev_ = ego_bev.view(self.embed_dims,self.bev_w,self.bev_h)
                bev_groups.append(ego_bev_)
                continue
            else:
                #Todo: there need to transfomation, and later will not need.
                op = self.sca(ego_bev, current_feat_flatten, current_feat_flatten,None, bev_pos, spatial_shapes, current_reference_points_cam, current_bev_mask,
                 level_start_index)
                neighbor_bev = self.tsblock(op)
                neighbor_bev_ = neighbor_bev.view(self.embed_dims,self.bev_w,self.bev_h)
                bev_groups.append(neighbor_bev_)
        bev_groups = torch.stack(bev_groups,dim=0)

        x_fuse = self.bevfusion(bev_groups,record_len,pairwise_t_matrix)

        psm = self.cls_head(x_fuse)
        rm = self.reg_head(x_fuse)
        output_dict = {'psm': psm,
                       'rm': rm}
        if self.use_dir:
            dm = self.dir_head(x_fuse)
            output_dict.update({"dm": dm})

        return output_dict




