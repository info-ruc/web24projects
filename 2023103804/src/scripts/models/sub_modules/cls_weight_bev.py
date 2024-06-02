from turtle import forward
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

def get_colla_feats(x, record_len, pairwise_t_matrix):
    # (B,L,L,2,3)
    _, C, H, W = x.shape
    B, L = pairwise_t_matrix.shape[:2]
    split_x = regroup(x, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
        # update each node i
        num_agents = batch_node_features[b].shape[0]
        neighbor_features = []
        for i in range(num_agents):
            # i = 0 # ego
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                                  t_matrix[i, :, :, :],
                                                  (H, W))
            neighbor_features.append(neighbor_feature.unsqueeze(0))
        neighbor_features = torch.cat(neighbor_features, dim=0)  # num_agents * num_agents * c * h * w
        out.append(neighbor_features)
    return out


def cls_weights(bev_cls_weight,record_len, pairwise_t_matrix,):
    bev_cls_weight = bev_cls_weight.permute(0,2,3,1)
    assert bev_cls_weight.shape[-1] == 1
    trans_cls_weights = get_colla_feats(bev_cls_weight, record_len, pairwise_t_matrix)
    bev_cls_map = torch.max(trans_cls_weights[0][0],dim=0)[0]
    bev_cls_map[bev_cls_map < 0.5] = 0
    bev_cls_map[bev_cls_map > 0.5] = 1
    return bev_cls_map


def get_t_matrix(pairwise_t_matrix, H, W, downsample_rate, discrete_ratio):
    # (B,L,L,2,3)
    pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]  # [B, L, L, 2, 3]
    pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
    pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
    pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0, 2] / (downsample_rate * discrete_ratio * W) * 2
    pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1, 2] / (downsample_rate * discrete_ratio * H) * 2
    return pairwise_t_matrix







