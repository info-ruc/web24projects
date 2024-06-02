import torch
import torch.nn as nn

from opencood.models.spatial_cross_attention import SpatialCrossAttention, SimpleNetwork
from opencood.models.sub_modules.fpn import FPN
from opencood.models.sub_modules.ref_3d import get_reference_points, point_sampling_handwrite
from opencood.utils.common_utils import get_true_lidar2img_matrix


class BEVencoder(nn.Module):
    def __init__(self,args):
        super(BEVencoder,self).__init__()
        # get feature map
        self.bev_h = args['bev_h']
        self.embed_dims = args['img_features_C']
        self.bev_w = args['bev_w']
        self.num_points_in_pillar = args['num_points_in_pillar']
        self.num_cams = args['Ncams']
        self.pc_range = args['gt_range']
        self.get_mul_fea = FPN(args)
        self.sca = SpatialCrossAttention().to('cuda')
        self.tsblock = SimpleNetwork(self.embed_dims,self.embed_dims)
        self.data_aug_conf = args['data_aug_conf']  # 数据增强配置参数

    def forward(self,data_dict,bev_query,bev_pos,cams_embeds,use_cams_embeds,level_embeds):
        image_inputs_dict = data_dict['image_inputs']
        record_len = data_dict['record_len']
        lidar2img_matrix = data_dict['lidar2img_matrix']
        x, rots, trans, intrins, post_rots, post_trans = \
            image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], \
                image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
        # B means agent_num    N means camera
        B, N, C, H, W = x.shape

        feat_flatten = []
        spatial_shapes = []

        x = x.view(-1,C,H,W)
        mlv_fea = self.get_mul_fea(x)

        for lvl, feat in enumerate(mlv_fea):
            bs, c, h, w = feat.shape
            feat = feat.unsqueeze(0).reshape(-1, N, c, h, w)
            spatial_shape = (h, w)
            # 变成 num_cam,bs,h*w,C (6,1,116*200,256)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if use_cams_embeds:
                # cams_embeds: (num_cas,bs,h*w,C)
                feat = feat + cams_embeds[:, None, None, :].to(feat.dtype)
            # level_embeds: 1 × 256
            feat = feat + level_embeds[None,
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

        # reference_points_cam, bev_mask = point_sampling(
        #     ref_3d, self.pc_range, lidar2img_matrix_true, H, W , rots, trans, intrins, post_rots, post_trans)
        ogfH, ogfW = self.data_aug_conf['final_dim']
        reference_points_cam, bev_mask = point_sampling_handwrite(
            ref_3d, self.pc_range, lidar2img_matrix_true, ogfH, ogfW , rots, trans, intrins, post_rots, post_trans)

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


        return bev_query, bev_groups