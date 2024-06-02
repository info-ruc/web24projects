from turtle import forward
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import seaborn as sns
import math
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from matplotlib import pyplot as plt
def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x
    
def uncertainty(depth_probs, u_thre=10):
    """
    Get the certainty mask whose entropy is lower than the threshold.

    Parameters
    ----------
    depth_probs : estimated depth distribution (b, num_agents, d, h, w)

    Returns
    -------
    uncertainty_mask : uncertainty mask (b, num_agents, h, w)
    """
    entropy = - depth_probs * torch.log(depth_probs + 1e-6)
    entropy = entropy.sum(dim=-3)
    # return entropy
    uncertainty_mask = (entropy < u_thre) * 1.0
    return uncertainty_mask

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
        neighbor_features = torch.cat(neighbor_features, dim=0) # num_agents * num_agents * c * h * w
        out.append(neighbor_features)
    return out

def cost_volume(feats, depth_probs, record_len, pairwise_t_matrix, masked_depth_probs=None, valid_depth_thre=0.1):
    """
    Get the consistency weights across multi agents' view.

    Parameters
    ----------
    feats: feature maps (b, num_agents, d, c, h, w)
    depth_probs : estimated depth distribution (b, num_agents, d, h, w)
    masked_depth_probs: estimated depth distribution filtered by uncertanty mask  (b, num_agents, d, h, w)
    pairwise_t_matrix: transformation matrix (b, num_agents, 3, 3)
    valid_depth_thre: depth threshold, filter out the inconfidence estimations (scalar)

    Returns
    -------
    updated_depth_probs: promoted depth distribution by consistency (b, num_agents, d, h, w)
    """
    # warp feature into ego agent's coordinate system
    val_feats = get_colla_feats(feats, record_len, pairwise_t_matrix)   # [(num_agents, num_agents, C,H,W), ...]
    if masked_depth_probs is None:
        val_probs = get_colla_feats(depth_probs, record_len, pairwise_t_matrix)
    else:
        val_probs = get_colla_feats(masked_depth_probs, record_len, pairwise_t_matrix)

    B = len(record_len)
    cost_v = []
    for b in range(B):
        # get consistency score
        val_feat = val_feats[b] # (num_agents,num_agents, c, h, w)
        val_prob = val_probs[b] # (num_agents,num_agents, h, w)
        num_agents = val_feat.shape[0]
        cost_a = []
        for i in range(num_agents):
            sim_score = (val_feat[i,i:i+1] * val_feat[i,:]).sum(dim=-3)    # (num_agents, h, w)
            binary_mask = (val_prob[i,:] > valid_depth_thre).squeeze(1)   # (num_agents, h, w)
            s = (sim_score * binary_mask).sum(dim=0).unsqueeze(0)  # (1, h, w)
            cost_a.append(s)
        cost_a = torch.cat(cost_a, dim=0).unsqueeze(1) # (num_agents, 1, h, w)
        cost_v.append(cost_a)
        
    cost_v = torch.cat(cost_v, dim=0) # (sum(record_len), 1, h, w)
    return cost_v

class CollaDepthNet(nn.Module):
    def __init__(self, dim=128, downsample_rate=1, discrete_ratio=1):
        super().__init__()
        self.downsample_rate = downsample_rate
        self.discrete_ratio = discrete_ratio
        self.norm = nn.LayerNorm(dim)
        self.depth_net = nn.Sequential(
                                    nn.Conv2d(2, 32,
                                        kernel_size=7, padding=3, bias=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 1,
                                        kernel_size=3, padding=1, bias=True))
    
    def get_t_matrix(self, pairwise_t_matrix, H, W, downsample_rate, discrete_ratio):
        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (downsample_rate * discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (downsample_rate * discrete_ratio * H) * 2
        return pairwise_t_matrix
    
    def forward(self, x, record_len, pairwise_t_matrix, depth_probs):
        # Multi-view matching
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1).flatten(0,2)
        x = self.norm(x)
        x = x.view(B, H, W, C).permute(0,3,1,2)

        pairwise_t_matrix = self.get_t_matrix(pairwise_t_matrix, H, W, downsample_rate=1, discrete_ratio=self.discrete_ratio)
        cost_v = cost_volume(x, depth_probs, record_len, pairwise_t_matrix, masked_depth_probs=None, valid_depth_thre=0.1) # (B, 1, H, W)
        updated_depth = torch.cat([depth_probs, cost_v], dim=1)
        updated_depth = self.depth_net(updated_depth)
        updated_depth = updated_depth.sigmoid()
        return updated_depth


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class mulclsdepthfusion(nn.Module):
    def __init__(self, downsample_rate=1, discrete_ratio=1, ntoken=3, d_model=256):
        super(mulclsdepthfusion, self).__init__()
        self.downsample_rate = downsample_rate
        self.feature_ex = FeatureExtractor()
        self.linear1 = nn.Linear(ntoken, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.discrete_ratio = discrete_ratio
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8),
            num_layers=3
        )
        self.linear2 = nn.Linear(d_model, ntoken)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def get_t_matrix(self, pairwise_t_matrix, H, W, downsample_rate, discrete_ratio):
        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]  # [B, L, L, 2, 3]
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
        pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0, 2] / (downsample_rate * discrete_ratio * W) * 2
        pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1, 2] / (downsample_rate * discrete_ratio * H) * 2
        return pairwise_t_matrix

    def forward(self, x, record_len, pairwise_t_matrix, depthbev, cls_bev):

        # data1 = cls_bev[0][1].detach().cpu().numpy()
        # data2 = cls_bev[0][0].detach().cpu().numpy()
        # data3 = depthbev[0][0].detach().cpu().numpy()
        # # 显示 data1
        # plt.figure()
        # sns.heatmap(data1, cmap="Reds")
        # plt.axis('off')
        # plt.savefig(f'{directory}fore', transparent=False, dpi=400)
        #
        # # 显示 data2
        # plt.figure()
        # sns.heatmap(data2, cmap="Reds")
        # plt.axis('off')
        # plt.savefig(f'{directory}background', transparent=False, dpi=400)

        # 显示 data3
        # plt.figure()
        # sns.heatmap(data3,cmap="viridis")
        # plt.axis('off')

        B, C, H, W = x.shape  # C = 128
        pairwise_t_matrix = self.get_t_matrix(pairwise_t_matrix, H, W, downsample_rate=1,
                                              discrete_ratio=self.discrete_ratio)

        # fuse depth with cls_img for a better
        depth_fuse_cat = torch.cat((depthbev, cls_bev), dim=1)

        depth_fuse = self.feature_ex(depth_fuse_cat)
        _, c, h, w = depth_fuse.shape
        # befor upsample , do dual attention for multi-agent fusion
        depth_fuse_list = get_colla_feats(depth_fuse, record_len,
                                          pairwise_t_matrix)  # [(num_agents, num_agents, C,H,W), ...]
        depth_fuse_all = depth_fuse_list[0]
        depth_fuse_end = []
        for i in range(B):
            depth_fuse_one = depth_fuse_all[i] # N,C,H,W
            depth_fuse_one = depth_fuse_one.permute(0,2,3,1).reshape(-1,1,c) # n*h*w,1,c
            depth_fuse_one = depth_fuse_one + self.positional_encoding(depth_fuse_one)
            depth_fuse_one = self.transformer_encoder(depth_fuse_one)
            depth_fuse_one = torch.max(depth_fuse_one.reshape(B,c,h,w),dim=0)[0]
            depth_fuse_end.append(depth_fuse_one)
        depth_fuse_end = torch.stack(depth_fuse_end)


        #TODO: check if we can use the same logical as the dual mask ?
        depth_fuse = self.upsample(depth_fuse_end)

        # vis
        # if iter%20 == 0:
        #     directory = f'/home/fulongtai/CoCa3D/opencood/figures/bev/{iter}/'
        #     data = depthbev[0][0].detach().cpu().numpy()
        #     plt.figure()
        #     sns.heatmap(data, cmap="viridis", cbar=False, xticklabels=False, yticklabels=False)
        #     plt.savefig(f"{directory}bevdepth")
        #     plt.figure()
        #     data = depth_fuse_cat[0].permute(1,2,0).detach().cpu().numpy()
        #     plt.imshow(data)
        #     plt.axis("off")
        #     plt.savefig(f"{directory}persudo-image")
        #
        #     #up
        #     t = normalize_tensor(depth_fuse)
        #     data1 = t[0][0].detach().cpu().numpy()
        #     plt.figure()
        #     sns.heatmap(data1, cmap="viridis", cbar=False, xticklabels=False, yticklabels=False)
        #     plt.savefig(f"{directory}dualattention-depth")
        #     plt.figure()
        #     data = t[0].permute(1,2,0).detach().cpu().numpy()
        #     plt.imshow(data,vmin=0,vmax=1,cmap='jet')
        #     plt.axis("off")
        #     plt.savefig(f"{directory}attention-image")


        # out_expanded = depth_fuse.unsqueeze(2)  # Shape becomes N x 3 x 1 x H x W
        # image_features_expanded = x.unsqueeze(1)  # Shape becomes N x 1 x C x H x W
        # # Element-wise multiplication, shape becomes N x 3 x C x H x W
        # fused_features = out_expanded * image_features_expanded
        # # Sum along the second dimension (indexed as 1), shape becomes N x C x H x W
        # # Essentially, the transformer output is used to weigh the original image features, potentially highlighting important areas or features for the task at hand.
        # fused_features_collapsed = fused_features.sum(dim=1)
        # fused_features_collapsed = x + fused_features_collapsed

        depth = depth_fuse[:,0:1,:,:]
        mask = depth_fuse[:,1:3,:,:]
        mask = torch.softmax(mask,dim=1)
        cls_nocar = mask[:,0:1,:,:]
        cls_car = mask[:,1:,:,:]
        fdepth = cls_nocar * depth + cls_car * depth

        fused_features_collapsed = fdepth * x

        return fused_features_collapsed



class clsdepthfusion(nn.Module):
    def __init__(self, downsample_rate=1, discrete_ratio=1 ,ntoken=3, d_model = 256):
        super(clsdepthfusion, self).__init__()
        self.downsample_rate = downsample_rate
        self.feature_ex = FeatureExtractor()
        self.linear1 = nn.Linear(ntoken, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.discrete_ratio = discrete_ratio
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8),
            num_layers=3
        )
        self.linear2 = nn.Linear(d_model, ntoken)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def get_t_matrix(self, pairwise_t_matrix, H, W, downsample_rate, discrete_ratio):
        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (downsample_rate * discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (downsample_rate * discrete_ratio * H) * 2
        return pairwise_t_matrix

    def forward(self, x, record_len, pairwise_t_matrix, depthbev, cls_bev):
        B, C, H, W = x.shape   # C = 128
        pairwise_t_matrix = self.get_t_matrix(pairwise_t_matrix, H, W, downsample_rate=1, discrete_ratio=self.discrete_ratio)

        # fuse depth with cls_img for a better
        depth_fuse_cat = torch.cat((depthbev,cls_bev),dim=1)
        #
        #
        # data = depth_fuse_cat[0].permute(1,2,0).detach().cpu().numpy()
        # plt.figure()
        # plt.imshow(data)
        # plt.axis('off')
        # plt.savefig('/home/fulongtai/CoCa3D/opencood/figures/bev/persudo-image', transparent=False, dpi=400)
        # plt.show()
        # plt.close()

        depth_fuse = self.feature_ex(depth_fuse_cat)
        n_agent,_,h,w = depth_fuse.shape
        depth_fuse = depth_fuse.flatten(2).permute(2, 0, 1)
        depth_fuse = depth_fuse + self.positional_encoding(depth_fuse)
        depth_fuse = self.transformer_encoder(depth_fuse)  # h*w , b , c
        depth_fuse = depth_fuse.permute(1, 2, 0).reshape(B, -1, h, w)  # Shape: [N, D, H, W]

        depth_fuse = self.upsample(depth_fuse)
        data = depth_fuse[0].permute(1,2,0).detach().cpu().numpy()
        plt.figure()
        plt.imshow(data)  # 假设data3是灰度图
        plt.axis('off')
        plt.savefig('/home/fulongtai/CoCa3D/opencood/figures/bev/dualattention-persudo-image', transparent=False, dpi=400)
        plt.show()
        plt.close()


        out_expanded = depth_fuse.unsqueeze(2)  # Shape becomes N x 3 x 1 x H x W
        image_features_expanded = x.unsqueeze(1)  # Shape becomes N x 1 x C x H x W
        # Element-wise multiplication, shape becomes N x 3 x C x H x W
        fused_features = out_expanded * image_features_expanded
        # Sum along the second dimension (indexed as 1), shape becomes N x C x H x W
        # Essentially, the transformer output is used to weigh the original image features, potentially highlighting important areas or features for the task at hand.
        fused_features_collapsed = fused_features.sum(dim=1)
        fused_features_collapsed = x + fused_features_collapsed

        return fused_features_collapsed


# Define feature extractor
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Downsample

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Downsample

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Downsample
            # ... (more layers as needed)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # Downsample
        )

    def forward(self, x):
        return self.layers(x)


def normalize_tensor(t):
    # t: input tensor of shape N,3,H,W
    min_vals = t.min(dim=-1)[0].min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
    max_vals = t.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)

    return (t - min_vals) / (max_vals - min_vals)




