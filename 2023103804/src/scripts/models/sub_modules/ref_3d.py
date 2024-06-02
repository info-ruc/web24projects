import torch
import numpy as np


def point_sampling_handwrite(reference_points, pc_range, l2i_tf, H, W, rots, trans, intrins, post_rots, post_trans):
    # 并不是使用lidar，而是使用雷达坐标系，其实和ego坐标系差不多
    # lidar2img = []
    # for img_meta in img_metas:
    #     # img_meta[0]['lidar2img'][0]-->4×4  一个变换矩阵从lidar坐标系转化到img坐标系。 最后一个0代表相机数
    #     lidar2img.append(img_meta['lidar2img'])
    # lidar2img = np.asarray()
    # lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)

    #  reference_points:  BS,1,40000,3
    B, N, _ = trans.shape  # B:4(batchsize)    N: 4(相机数目)

    reference_points = reference_points.clone()
    # # 抵消数据增强及预处理对像素的变化
    # reference_points = reference_points - post_trans.view(B, N, 1, 3)
    # reference_points = torch.inverse(post_rots).view(B, N, 1, 3, 3).matmul(reference_points.unsqueeze(-1))
    # 反归一化。 我们需要将坐标从[0,1]区间反归一化到它们在实际空间中的值，  最后加的意思是把中心移到自车上

    reference_points[..., 0:1] = reference_points[..., 0:1] * \
                                 (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * \
                                 (pc_range[4] - pc_range[1]) + pc_range[1]
    # pc_range[5] - pc_range[2]=8
    reference_points[..., 2:3] = reference_points[..., 2:3] * \
                                 (pc_range[5] - pc_range[2]) + pc_range[2]
    # 把ref_points补了一个1，变成齐次，和lidar2img的变换矩阵对齐
    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)
    num_cam = l2i_tf.size(2)  # 4
    n_agent = l2i_tf.size(1)  # unknow
    reference_points = reference_points.permute(1, 0, 2, 3)
    D, B, num_query = reference_points.size()[:3]
    reference_points = reference_points.view(
        D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
    l2i_tf = l2i_tf.reshape(1,B,N,1,4,4).repeat(D,1,1,num_query,1,1)
    reference_points_cam = torch.matmul(l2i_tf.to(torch.float32),
                                        reference_points.to(torch.float32)).squeeze(-1)
    eps = 1e-5
    # 深度在相机后面的 点 去掉 // 就是对投影完的点进行筛选，很多点在6个camera中不会都出现。
    bev_mask = (reference_points_cam[..., 2:3] > eps)
    # print(torch.sum(bev_mask.view(-1), dim=-1))
    # 透视投影。 将3D空间中的点投影到2D图像平面上。这是通过将3D点的x和y坐标除以其z坐标（即深度）来实现的。
    # 这个操作基于透视投影的基本原理，即一个物体离观察者越远，其在观察者视野中的投影就越小。
    # 得到像素坐标,u,v
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

    # reference_points_cam = reference_points_cam[..., 0:2] / reference_points_cam[..., 2:3]
    # 这段代码是对投影到2D图像平面上的点进行归一化处理。这是通过将点的x和y坐标除以图像的宽度和高度来实现的。
    reference_points_cam[..., 0] /= W
    reference_points_cam[..., 1] /= H
    # 确保归一化之后 参考点在0-1之间 ，过滤掉了很多没有必要的点
    bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1)
                & (reference_points_cam[..., 0:1] < 1)
                & (reference_points_cam[..., 0:1] > 0.0))

    bev_mask = bev_mask.new_tensor(
        np.nan_to_num(bev_mask.cpu().numpy()))

    reference_points_cam = reference_points_cam.squeeze(2)
    bev_mask = bev_mask.squeeze(2)

    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    # reference_points_cam：6,1,,40000,4,2    bev_mask:6,1,40000,4
    return reference_points_cam, bev_mask




def point_sampling(reference_points, pc_range,  l2i_tf, H,W,rots, trans, intrins, post_rots, post_trans):
    # 并不是使用lidar，而是使用雷达坐标系，其实和ego坐标系差不多
    # lidar2img = []
    # for img_meta in img_metas:
    #     # img_meta[0]['lidar2img'][0]-->4×4  一个变换矩阵从lidar坐标系转化到img坐标系。 最后一个0代表相机数
    #     lidar2img.append(img_meta['lidar2img'])
    # lidar2img = np.asarray()
    # lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)

    #  reference_points:  BS,1,40000,3
    B, N, _ = trans.shape  # B:4(batchsize)    N: 4(相机数目)

    reference_points = reference_points.clone()
    # 抵消数据增强及预处理对像素的变化
    # reference_points = reference_points - post_trans.view(B, N, 1, 3)
    # reference_points = torch.inverse(post_rots).view(B, N, 1, 3, 3).matmul(reference_points.unsqueeze(-1))
    # 反归一化。 我们需要将坐标从[0,1]区间反归一化到它们在实际空间中的值，  最后加的意思是把中心移到自车上

    reference_points[..., 0:1] = reference_points[..., 0:1] * \
        (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * \
        (pc_range[4] - pc_range[1]) + pc_range[1]
    # pc_range[5] - pc_range[2]=8
    reference_points[..., 2:3] = reference_points[..., 2:3] * \
        (pc_range[5] - pc_range[2]) + pc_range[2]
    # 把ref_points补了一个1，变成齐次，和lidar2img的变换矩阵对齐

    reference_points = reference_points.squeeze(-1)
    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)

    dtype = l2i_tf.type()
    #
    transform_matrix_np = np.array(
        [[0, 0, 1, 0],
         [1, 0, 0, 0],
         [0, -1, 0, 0],
         [0, 0, 0, 1]],
        dtype=np.float32)
    UE_OPENCV_inv = np.linalg.inv(transform_matrix_np)
    UE_OPENCV_inv = torch.from_numpy(UE_OPENCV_inv).to('cuda')

    num_cam = l2i_tf.size(2)  # 4
    n_agent = l2i_tf.size(1)  # unknow
    reference_points_cam_group = []
    l2i_tf = l2i_tf @ UE_OPENCV_inv.type(dtype)
    for n in range(n_agent):
        reference_points_one = reference_points[n:n+1].permute(1, 0, 2, 3)
        # in this part , B = N_agent * batch_szie , D = num_points_pillar
        D, B, num_query = reference_points_one.size()[:3]

        reference_points_one = reference_points_one.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img_one = l2i_tf[:,n:n+1,:,:,:]
        lidar2img_one = lidar2img_one.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        # lidar2img = l2i_tf.view(
        #     1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
        # (4,4) * [x,y,z,1] --> (zc*u,zc*v,zc,1)

        reference_points_one = (lidar2img_one.to(torch.float32)  @ reference_points_one.to(torch.float32)).squeeze(-1)


        # reference_points_one = reference_points_one - trans[n:n+1].view(1, B, num_cam, 1, 1, 3)
        # combine = intrins.matmul(torch.inverse(rots))
        # reference_points_one = combine.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points_one).squeeze(-1)
        reference_points_cam_group.append(reference_points_one)
    # D, num_agent, num_camera , 1 , 4 ,40000, 4
    reference_points_cam_group = torch.stack(reference_points_cam_group,dim=1).squeeze(2)
    reference_points_cam = reference_points_cam_group
    eps = 1e-5
    # 深度在相机后面的 点 去掉 // 就是对投影完的点进行筛选，很多点在6个camera中不会都出现。
    bev_mask = (reference_points_cam[..., 2:3] > eps)
    print(torch.sum(bev_mask.view(-1),dim=-1))
    # 透视投影。 将3D空间中的点投影到2D图像平面上。这是通过将3D点的x和y坐标除以其z坐标（即深度）来实现的。
    # 这个操作基于透视投影的基本原理，即一个物体离观察者越远，其在观察者视野中的投影就越小。
    # 得到像素坐标,u,v
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * -eps)

    # reference_points_cam = reference_points_cam[..., 0:2] / reference_points_cam[..., 2:3]

    # 这段代码是对投影到2D图像平面上的点进行归一化处理。这是通过将点的x和y坐标除以图像的宽度和高度来实现的。
    reference_points_cam[..., 0] /= W
    reference_points_cam[..., 1] /= H
    # 确保归一化之后 参考点在0-1之间 ，过滤掉了很多没有必要的点
    bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1)
                & (reference_points_cam[..., 0:1] < 1)
                & (reference_points_cam[..., 0:1] > 0.0))

    bev_mask = bev_mask.new_tensor(
        np.nan_to_num(bev_mask.cpu().numpy()))

    reference_points_cam = reference_points_cam.squeeze(2)
    bev_mask = bev_mask.squeeze(2)

    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    # reference_points_cam：6,1,,40000,4,2    bev_mask:6,1,40000,4
    return reference_points_cam, bev_mask


def get_reference_points(H, W, Z=4, num_points_in_pillar=4, bs=1, device='cuda', dtype=torch.float):
    """Get the reference points used in SCA and TSA.
    Args:
        H, W: spatial shape of bev.
        Z: hight of pillar.
        D: sample D points uniformly from each pillar.
        device (obj:`device`): The device where
            reference_points should be.
    Returns:
        Tensor: reference points used in decoder, has \
            shape (bs, num_keys, num_levels, 2).
    """

    # reference points in 3D space, used in spatial cross-attention (SCA)
    # 相较于没有高度的2D，3D是在z上取4个点，每个200*200的位置都有
    # 4个点

    zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                        device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
    xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                        device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
    ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                        device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
    ref_3d = torch.stack((xs, ys, zs), -1)
    ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
    ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
    # ref_3d: 1，4，40000，3  --> 4个点，200*200，3代表x,y,z坐标

    return ref_3d