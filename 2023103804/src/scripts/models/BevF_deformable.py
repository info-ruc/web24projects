
import torch
import math
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from opencood.models.sub_modules.BEVDecoder import DeformableTransformerDecoder, DeformableTransformerDecoderLayer, \
    inverse_sigmoid
from opencood.models.sub_modules.detr_module import PositionEmbeddingSine
from opencood.models.sub_modules.lss_submodule import BevEncodeMSFusion, mulBEVfusion
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from matplotlib import pyplot as plt

from opencood.models.spatial_cross_attention import SpatialCrossAttention,  SimpleNetwork
from opencood.models.sub_modules.fpn import FPN
from opencood.models.sub_modules.ref_3d import *

from opencood.models.sub_modules.BEVEncoder import BEVencoder


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

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



class BevFdeformable(nn.Module):
    def __init__(self, args):
        super(BevFdeformable, self).__init__()
        # any para will be put into yaml
        fusion_args = args['fusion_args']
        self.outC = args['outC']
        self.embed_dims = args['img_features_C']
        self.downsample = args['img_downsample']  # 下采样倍数
        self.num_cams = args['Ncams']
        self.bev_h = args['bev_h']
        self.bev_w = args['bev_w']
        self.num_feature_levels = args['num_feature_levels']
        # bev_query
        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims)
        self.query_embedding = nn.Embedding(900,
                                            self.embed_dims * 2)

        self.use_cams_embeds = args['use_cams_embeds']
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2)

        self.cycle = args['cycle']
        self.discrete_ratio = fusion_args['args']['voxel_size'][0]
        self.downsample_rate = 1

        self.encoder = BEVencoder(args)

        self.Decoderlayer = DeformableTransformerDecoderLayer()
        self.decoder = DeformableTransformerDecoder(self.Decoderlayer,4)

        #object query
        self.query_embed = nn.Embedding(100, self.embed_dims * 2)

        # self.bevfusion = BevFusion(inC=self.camC, outC=self.outC)

        self.bevfusion = mulBEVfusion(fusion_args)

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        self.cls_head = nn.Conv2d(self.outC, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.outC, 7 * args['anchor_number'],
                                  kernel_size=1)
        self.class_embed = nn.Linear(self.embed_dims, 2)
        self.bbox_embed = MLP(self.embed_dims, self.embed_dims, 4, 3)

        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        self.class_embed = nn.ModuleList([self.class_embed for _ in range(1)])
        self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(1)])
        if 'dir_args' in args.keys():


            self.use_dir = True
            self.dir_head = nn.Conv2d(self.outC, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
        else:
            self.use_dir = False

        self.initial_para()

    def initial_para(self):
        init.normal_(self.cams_embeds)
        init.normal_(self.level_embeds)
        nn.init.xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        nn.init.constant_(self.reference_points.bias.data, 0.)


    def forward(self,data_dict):
        image_inputs_dict = data_dict['image_inputs']
        record_len = data_dict['record_len']
        lidar2img_matrix = data_dict['lidar2img_matrix']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        x, rots, trans, intrins, post_rots, post_trans = \
            image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], \
                image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
        # B means agent_num    N means camera
        B, N, C, H, W = x.shape

        x = x.view(-1,C,H,W)
        # pre bev_query.weight
        bev_query = self.bev_embedding.weight
        # bev_mask = torch.zeros(bs, bev_h, bev_w)
        # we only need one bev
        bev_query = bev_query.unsqueeze(1)
        bev_pos = positionalencoding2d(self.embed_dims,self.bev_h,self.bev_w)
        bev_pos = bev_pos.permute(1, 2, 0)
        bev_pos = bev_pos.view(1, -1, self.embed_dims)
        bev_pos = bev_pos.to('cuda')

        object_query_embeds = self.query_embedding.weight

        # bev_embed shape: n_agent, bev_h*bev_w, embed_dims
        bev_embed, bev_groups = self.encoder(data_dict,bev_query,bev_pos,self.cams_embeds,self.use_cams_embeds,self.level_embeds)

        # query_embed, tgt = torch.split(query_embed, c, dim=1)
        # query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        # tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        # reference_points = self.reference_points(query_embed).sigmoid()
        # init_reference_out = reference_points


        query_embeds = self.query_embed.weight
        bs = 1
        query_pos, query = torch.split(
            query_embeds, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference = reference_points

        # decoder
        hs, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=None,
            cls_branches=None,
            spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),)

        # output = self.decoder(object_query_embeds,)

        # B * N
        x_fuse = self.bevfusion(bev_groups,record_len,pairwise_t_matrix)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference.squeeze(0)
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        psm = self.cls_head(x_fuse)
        rm = self.reg_head(x_fuse)
        output_dict = {'psm': psm,
                       'rm': rm}
        if self.use_dir:
            dm = self.dir_head(x_fuse)
            output_dict.update({"dm": dm})
        # output_dict = {'psm': outputs_class[-1],
        #                'rm': outputs_coord[-1]}
        # if self.use_dir:
        #     dm = self.dir_head(x_fuse)
        #     output_dict.update({"dm": dm})


        return output_dict




