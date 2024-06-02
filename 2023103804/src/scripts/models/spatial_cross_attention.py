# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Kang Yang
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def device_select(x,y):
    if x.device != y.device:
        if 'cuda' in str(x.device):
            y = y.to('cuda')
        else:
            x = x.to('cuda')
    return x,y

class SpatialCrossAttention(nn.Module):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=4,
                 pc_range=None,
                 dropout=0.1,):
        super(SpatialCrossAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformableAttention3D()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        nn.init.xavier_uniform(self.output_proj.weight)

    def forward(self,
                query,  # bev query
                key,    # feature map --> feat_flatten
                value,    # the same as key
                residual=None,
                query_pos=None,  # PE
                spatial_shapes=None, # spatial_shapes get
                reference_points_cam=None, # get
                bev_mask=None, # get
                level_start_index=None, # get
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        # query: 1, 40000, 256
        # key:6,30825,1,256   就是不同层级的feature map的一维大小叠加 (4层)
        # reference_points_cam：6,1,,40000,4,2    bev_mask:6,1,40000,4
        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            inp_residual = query
        slots = torch.zeros_like(query)

        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)
        indexes = []
        # agent_ihdexes = []
        # num_points,n_agent,_,_ = bev_mask.shape
        # bev_mask = bev_mask.permte(1,0,2,3,4)
        # for mask_per_agent in bev_mask:
        #     for j,mask_per_img in enumerate(mask_per_agent):
        #         index_query_per_img = mask_per_img.sum(-1).nonzero().squeeze(-1)
        #         indexes.append(index_query_per_img)
        for i, mask_per_img in enumerate(bev_mask):
            # bev_mask:6,1,40000,4
            # bev_mask[0][0].sum(-1)，也就是对于第一个相机的第一个bs，40000个点，其中有效点的个数
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])
        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        # queries_rebatch.shape [1,6,9675,256] 某一趟对于每个相机，最大有效点个数是9675
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        # reference_points_rebatch.shape [1,6,9675,4,2]
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])
        # asset query and per_img has the same device
        query,reference_points_cam = device_select(query,reference_points_cam)
        for j in range(bs):
            # 遍历相机
            for i, reference_points_per_img in enumerate(reference_points_cam):

                index_query_per_img = indexes[i]
                # query上，对应的索引，放到第j个bs第i个相机的 0~ len(index_query_per_img)
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        # 得到的queries.shape:1,6,9675,256
        queries = self.deformable_attention(query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
                                            key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len,
                                                                                           D, 2),
                                            spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len,
                                                                                      self.embed_dims)
        # 对于得到的queries，还要复原到BEV空间中，因为你 的 6× 9675只是部分  ，跟之前根据index找BEV对应位置刚好相反
        slots = slots.to('cuda')
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j,  index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]

        # slots = slots.to('cpu')
        slots = self.output_proj(slots)
        # slots 又被拍成了BEV query
        return self.dropout(slots)+ inp_residual


class MSDeformableAttention3D(torch.nn.Module):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=3,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super(MSDeformableAttention3D, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight, val=0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        # nn.init.xavier_uniform_(self.output_proj)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        # query是筛选之后的东西
        # value是ml feature map flatten之后的 东西
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        # sampling_offsets.shape:6,9675,8,4,8,2     为什么num_points=8？ 下面给出解答
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        # 点的概率分布
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)
        # attention_weights.shape: 6,9675,8,4,8
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        # reference_points  6，9675，4，2
        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                               offset_normalizer[None, None, None, :, None, :]
            # 这里的bs不是batch_size,而是camera_num
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            # num_Z_anchors 就是，一个ref_points 点，对应2个 cam上的点，
            # sampling_offsets.shape: 6,9675,8,4,2,4,2，分别表示 6个cam，9675个点（最大，不一定全是有效数据），8个head，
            # 4个特征图，2是2个anchors，4是4个bev的点，最后是2个xy坐标
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)

            sampling_offsets,reference_points = device_select(sampling_offsets,reference_points)

            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #
        value = value.to('cuda')
        spatial_shapes = spatial_shapes.to('cuda')
        sampling_locations = sampling_locations.to('cuda')
        attention_weights = attention_weights.to('cuda')
        output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        return output




class AddNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(AddNorm, self).__init__()
        self.norm = nn.LayerNorm(size, eps=eps)

    def forward(self, x, sublayer_output):
        "Add & Normalize"
        return self.norm(x + sublayer_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class SimpleNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(SimpleNetwork, self).__init__()
        self.add_norm1 = AddNorm(d_model)
        self.add_norm2 = AddNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = self.add_norm1(x, self.ff(x))
        x = self.add_norm2(x, self.ff(x))
        return x


if __name__ == '__main__':
    # Initialize and test
    d_model = 512
    d_ff = 2048
    x = torch.rand(1, 10, d_model)  # Batch size of 1, sequence length of 10

    model = SimpleNetwork(d_model, d_ff)
    output = model(x)
    print(output.shape)  # Should be [1, 10, 512]