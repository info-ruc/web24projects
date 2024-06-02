# testing multiview camera dataset

"""
pure camera api, remove codes of LiDAR
"""
import matplotlib.pyplot as plt
from locale import str
from builtins import enumerate, len, list
from PIL import Image
import math
from collections import OrderedDict
import cv2
import numpy as np
import torch
from PIL import Image
from icecream import ic
import pickle as pkl
import open3d as o3d
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import camera_basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
    gen_dx_bx,
    load_camera_data,
    lidar2img,
    coord_3d_to_2d, depth_transform
)
import seaborn as sns
from opencood.utils.transformation_utils import x1_to_x2, x_to_world
from opencood.utils.common_utils import read_json
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.visualization.simple_plot3d import canvas_bev


class depthCameraIntermediateFusionDataset(camera_basedataset.CameraBaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    detection outputs to ego.
    """

    def __init__(self, params, visualize, train=True):
        super(depthCameraIntermediateFusionDataset, self).__init__(params, visualize, train)
        self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.anchor_box = self.post_processor.generate_anchor_box()
        self.anchor_box_torch = torch.from_numpy(self.anchor_box)
        self.num_cam = params["fusion"]["args"]["data_aug_conf"]["Ncams"]
        if self.preload and self.preload_worker_num:
            self.retrieve_all_base_data_mp()
        elif self.preload:
            self.retrieve_all_base_data()

    def get_item_single_car_camera(self, selected_cav_base, ego_cav_base):
        """
        Process a single CAV's information for the train/test pipeline.


        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
            including 'params', 'camera_data'
        ego_pose : list, length 6
            The ego vehicle lidar pose under world coordinate.
        ego_pose_clean : list, length 6
            only used for gt box generation

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """

        transformation_matrix_list = []

        selected_cav_processed = {}
        ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']

        # calculate the transformation matrix
        transformation_matrix = \
            x1_to_x2(selected_cav_base['params']['lidar_pose'],
                     ego_pose)  # T_ego_cav
        transformation_matrix_clean = \
            x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                     ego_pose_clean)

        # generate targets label single GT
        visibility_map = np.asarray(cv2.cvtColor(selected_cav_base["bev_visibility.png"], cv2.COLOR_BGR2GRAY))
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
            [selected_cav_base], selected_cav_base['params']['lidar_pose'], visibility_map
        )

        # d_img= agents_image_inputs[j]['imgs'][n].numpy()
        # plt.imshow(d_img.transpose(1, 2, 0))
        # plt.axis('off')  # 不显示坐标轴
        # plt.show()


        # only use lidar to get depth infor
        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        projected_lidar = \
            box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                     transformation_matrix)
        if self.proj_first:
            lidar_np[:, :3] = projected_lidar
        # Need cloud points
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])

        # 设置参数
        x_min, x_max, grid_size_x = -48, 48, 0.4
        y_min, y_max, grid_size_y = -48, 48, 0.4

        # 计算网格大小
        grid_shape_x = int((x_max - x_min) / grid_size_x)
        grid_shape_y = int((y_max - y_min) / grid_size_y)

        # 使用 np.histogram2d 计算每个网格中的点的 z 坐标之和
        hist, xedges, yedges = np.histogram2d(
            lidar_np[:, 0], lidar_np[:, 1],
            bins=[grid_shape_x, grid_shape_y],
            range=[[x_min, x_max], [y_min, y_max]],
            weights=lidar_np[:, 2]
        )

        # 使用 np.histogram2d 计算每个网格中的点总数，用于计算平均高度
        count, _, _ = np.histogram2d(
            lidar_np[:, 0], lidar_np[:, 1],
            bins=[grid_shape_x, grid_shape_y],
            range=[[x_min, x_max], [y_min, y_max]]
        )
        safe_count = np.where(count > 0, count, 1)
        lidar_bev = np.where(count > 0, hist / safe_count, 0)

        # plt.figure(figsize=(10, 10))
        # plt.imshow(lidar_bev.T, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='jet')
        # plt.colorbar(label='Height')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')
        # plt.title('Average Height Grid')
        # plt.show()

        selected_cav_processed.update({
            'lidar_np_3d':lidar_bev
        })
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(lidar_np[:,:3])
        # o3d.visualization.draw_geometries([point_cloud])

        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
        )
        selected_cav_processed.update({"single_label_dict": label_dict})

        # adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/data.py
        camera_data_list = selected_cav_base["camera_data"]

        params = selected_cav_base["params"]
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        lidar_depth = []
        # generate the bounding box(n, 7) under the cav's space
        visibility_map = np.asarray(cv2.cvtColor(selected_cav_base["bev_visibility.png"], cv2.COLOR_BGR2GRAY))
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
            [selected_cav_base], selected_cav_base['params']['lidar_pose'], visibility_map
        )
        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
        )
        selected_cav_processed.update({"single_label_dict": label_dict})

        for idx, img in enumerate(camera_data_list):
            camera_coords = np.array(params["camera%d" % idx]["cords"]).astype(
                np.float32
            )

            ego_camera = x1_to_x2(params['lidar_pose'],camera_coords)
            ego_camera = np.array(
                [[0,1,0,0], [0, 0, -1, 0], [1, 0,  0, 0], [0, 0, 0, 1]],
                dtype=np.float32) @ ego_camera
            rot_1 = torch.from_numpy(
                ego_camera[:3, :3]
            )
            tran_1 = torch.from_numpy(ego_camera[:3, 3])  # T_wc

            camera_to_lidar = x1_to_x2(
                camera_coords, params["lidar_pose_clean"]
            ).astype(np.float32)  # T_LiDAR_camera
            camera_to_lidar = camera_to_lidar @ np.array(
                [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                dtype=np.float32)  # UE4 coord to opencv coord
            # lidar_to_camera = np.array(params['camera%d' % idx]['extrinsic']).astype(np.float32) # Twc^-1 @ Twl = T_camera_LiDAR
            camera_intrinsic = np.array(params["camera%d" % idx]["intrinsic"]).astype(
                np.float32
            )

            intrin = torch.from_numpy(camera_intrinsic)
            rot = torch.from_numpy(
                camera_to_lidar[:3, :3]
            )  # R_wc, we consider world-coord is the lidar-coord
            tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            img_src = [img]

            # depth
            if self.use_gt_depth:
                depth_img = selected_cav_base["depth_data"][idx]
                img_src.append(depth_img)
            else:
                depth_img = None

            # 2d foreground mask
            if self.use_fg_mask:
                _, _, fg_mask = coord_3d_to_2d(
                    box_utils.boxes_to_corners_3d(object_bbx_center[:len(object_ids)],
                                                  self.params['postprocess']['order']),
                    camera_intrinsic,
                    camera_to_lidar,
                    mask='float'
                )
                fg_mask = np.array(fg_mask * 255, dtype=np.uint8)
                fg_mask = Image.fromarray(fg_mask)
                img_src.append(fg_mask)

            point_depth = self.get_lidar_depth(rot_1,tran_1,intrin,img,lidar_np)
            # point_depth_1 = self.get_lidar_depth_1(camera_to_lidar,intrin,lidar_np)
            # data augmentation
            resize, resize_dims, crop, flip, rotate = sample_augmentation(
                self.data_aug_conf, self.train
            )

            point_depth_augmented = depth_transform(
                point_depth, resize, self.data_aug_conf['final_dim'],
                crop, flip, rotate)
            # plt.imshow(img)
            # plt.show()
            # data = point_depth_augmented.numpy()
            # sns.heatmap(data, cmap="viridis", cbar=True, vmin=0, vmax=50, xticklabels=80, yticklabels=80)
            # plt.show()
            # plt.close()

            img_src, post_rot2, post_tran2 = img_transform(
                img_src,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # decouple RGB and Depth

            img_src[0] = normalize_img(img_src[0])
            if self.use_gt_depth:
                img_src[1] = img_to_tensor(img_src[1]) * 255
            if self.use_fg_mask:
                img_src[-1] = img_to_tensor(img_src[-1])


            imgs.append(torch.cat(img_src, dim=0))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            lidar_depth.append(point_depth_augmented)
            transformation_matrix_tp = torch.from_numpy(transformation_matrix)
            transformation_matrix_list.append(transformation_matrix_tp)
        selected_cav_processed.update(
            {
                "image_inputs":
                    {
                        "imgs": torch.stack(imgs),  # [Ncam, 3or4, H, W]
                        "intrins": torch.stack(intrins),
                        "rots": torch.stack(rots),
                        "trans": torch.stack(trans),
                        "post_rots": torch.stack(post_rots),
                        "post_trans": torch.stack(post_trans),
                        "lidar_depth":torch.stack(lidar_depth),
                        'transformation_matrix': torch.stack(transformation_matrix_list)
                    }
            }
        )

        # generate targets label single GT
        visibility_map = np.asarray(cv2.cvtColor(ego_cav_base["bev_visibility.png"], cv2.COLOR_BGR2GRAY))
        object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
            [selected_cav_base], ego_cav_base['params']['lidar_pose'], visibility_map
        )
        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
        )

        selected_cav_processed.update(
            {
                "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                "object_bbx_mask": object_bbx_mask,
                "object_ids": object_ids,
                'transformation_matrix': transformation_matrix,
                'transformation_matrix_clean': transformation_matrix_clean
                # 'lidar_np': lidar_np
            }
        )

        # from opencood.utils import camera_utils
        # rand_idx = np.random.randint(0,10000)
        # for idx in range(4):
        #     camera_coords = np.array(params["camera%d" % idx]["cords"]).astype(
        #         np.float32)
        #     camera_to_lidar = x1_to_x2(
        #         camera_coords, params["lidar_pose_clean"]
        #     ).astype(np.float32)  # T_LiDAR_camera
        #     camera_to_lidar = camera_to_lidar @ np.array(
        #         [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        #         dtype=np.float32)  # UE4 coord to opencv coord
        #     # lidar_to_camera = np.array(params['camera%d' % idx]['extrinsic']).astype(np.float32) # Twc^-1 @ Twl = T_camera_LiDAR
        #     camera_intrinsic = np.array(params["camera%d" % idx]["intrinsic"]).astype(
        #         np.float32
        #     )
        #     camera_utils.coord_3d_to_2d(
        #         box_utils.boxes_to_corners_3d(object_bbx_center[:len(object_ids)], self.params['postprocess']['order']), \
        #         camera_intrinsic,\
        #         camera_to_lidar,\
        #         mask='float',\
        #         image=camera_data_list[idx], idx=rand_idx+idx
        #     )

        # filter lidar, visualize
        if self.visualize:
            # filter lidar
            lidar_np = selected_cav_base['lidar_np']
            lidar_np = shuffle_points(lidar_np)
            # remove points that hit itself
            lidar_np = mask_ego_points(lidar_np)
            # project the lidar to ego space
            # x,y,z in ego space
            projected_lidar = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                         transformation_matrix)
            selected_cav_processed.update({'projected_lidar': projected_lidar})

        return selected_cav_processed

    def get_images(self, idx):
        base_data_dict = self.retrieve_base_data(idx)
        camera_data_dict = OrderedDict()
        for cav_id, selected_cav_base in base_data_dict.items():
            camera_data_list = selected_cav_base["camera_data"]
            camera_data_dict[cav_id] = camera_data_list
        return camera_data_dict

    def __getitem__(self, idx):
        # get image,bev_visi, parameters(ego_pose,...)
        base_data_dict = self.retrieve_base_data(idx)

        base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        ego_id = -1
        ego_lidar_pose = []
        ego_cav_base = None

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                ego_lidar_pose_clean = cav_content['params']['lidar_pose_clean']
                ego_cav_base = cav_content
                break

        assert cav_id == list(base_data_dict.keys())[
            0], "The first element in the OrderedDict must be ego"
        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        agents_image_inputs = []
        object_stack = []
        object_id_stack = []
        single_label_list = []
        too_far = []
        lidar_pose_list = []
        lidar_pose_clean_list = []
        projected_lidar_clean_list = []
        lidar = []
        cav_id_list = []

        if self.visualize:
            projected_lidar_stack = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            # check if the cav is within the communication range with ego
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0]) ** 2 + (
                                  selected_cav_base['params'][
                                      'lidar_pose'][1] - ego_lidar_pose[
                                      1]) ** 2)

            # if distance is too far, we will just skip this agent
            if distance > self.params['comm_range']:
                too_far.append(cav_id)
                continue

            lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
            lidar_pose_list.append(selected_cav_base['params']['lidar_pose'])  # 6dof pose
            cav_id_list.append(cav_id)
        lidar_list = []
        for i, cav_id in enumerate(cav_id_list):
            selected_cav_base = base_data_dict[cav_id]
            # for get lidar
            selected_cav_processed = self.get_item_single_car_camera(
                selected_cav_base,
                ego_cav_base)

            # get depth from ego lidar
            # if i == 0:
            #     point_depth = self.get_lidar_depth(selected_cav_processed)
            lidar_list.append(torch.from_numpy(selected_cav_processed['lidar_np_3d']))
            object_stack.append(selected_cav_processed['object_bbx_center'])

            object_id_stack += selected_cav_processed['object_ids']

            agents_image_inputs.append(
                selected_cav_processed['image_inputs'])

            if self.visualize:
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])

            if self.supervise_single:
                single_label_list.append(selected_cav_processed['single_label_dict'])
        ########## Added by Yifan Lu 2022.10.10 ##############
        # generate single view GT label
        lidar_list = torch.stack(lidar_list)
        if self.supervise_single:
            single_label_dicts = self.post_processor.collate_batch(single_label_list)
            processed_data_dict['ego'].update(
                {"single_label_dict_torch": single_label_dicts}
            )

        ########## Added by Yifan Lu 2022.4.5 ################
        # filter those out of communicate range
        # then we can calculate get_pairwise_transformation
        for cav_id in too_far:
            base_data_dict.pop(cav_id)

        pairwise_t_matrix = \
            self.get_pairwise_transformation(base_data_dict,
                                             self.max_cav)
        ########## Added by Kang Yang 2023.9.14 ################
        # to get the lidar2img_matrix which will be used in spatial cross attention
        lidar2img_matrix = lidar2img(base_data_dict, self.max_cav, self.num_cam)

        ########## ############################ ################
        lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
        lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
        ######################################################

        # exclude all repetitive objects
        unique_indices = \
            [object_id_stack.index(x) for x in set(object_id_stack)]
        object_stack = np.vstack(object_stack)
        object_stack = object_stack[unique_indices]

        # make sure bounding boxes across all frames have the same number
        object_bbx_center = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        mask = np.zeros(self.params['postprocess']['max_num'])
        object_bbx_center[:object_stack.shape[0], :] = object_stack
        mask[:object_stack.shape[0]] = 1

        # merge preprocessed features from different cavs into the same dict
        cav_num = len(agents_image_inputs)

        merged_image_inputs_dict = self.merge_features_to_dict(agents_image_inputs, merge='stack')

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=self.anchor_box,
                mask=mask)

        processed_data_dict['ego'].update(
            {'object_bbx_center': object_bbx_center,
             'object_bbx_mask': mask,
             'object_ids': [object_id_stack[i] for i in unique_indices],
             'image_inputs': merged_image_inputs_dict,
             'label_dict': label_dict,
             'cav_num': cav_num,
             'pairwise_t_matrix': pairwise_t_matrix,
             'lidar2img_matrix': lidar2img_matrix,
             'lidar_poses_clean': lidar_poses_clean,
             'lidar_3d': lidar_list,
             'lidar_poses': lidar_poses})

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar':
                np.vstack(
                    projected_lidar_stack)})

        processed_data_dict['ego'].update({'sample_idx': idx,
                                           'cav_id_list': cav_id_list})

        return processed_data_dict


    # add by KangYang 2023.9.24
    def get_lidar_depth(self, rot,tran,intrins,img,lidar):
        '''
        rot:3,3
        tran:3
        intrins:3,3
        img: PngImageFile
        '''
        # rot_inv = torch.inverse(rot)
        #
        # lidar_tensor = torch.from_numpy(lidar[:,:3])
        # n_points,_= lidar_tensor.shape
        # tran = tran.view(1,-1).repeat(n_points,1)
        # rot_inv = rot_inv.repeat(n_points,1,1)
        # lidar_tensor = lidar_tensor - tran
        # # ext_matrix = np.linalg.inv(ext_matrix)[:3,:4]
        # # img_pts = (ext_matrix @ xyz_hom.T).T
        # rotated_point_cloud = torch.einsum('ijk,ik->ij', rot_inv, lidar_tensor)
        # depths = rotated_point_cloud[..., 2:3]
        # coloring = depths
        lidar_tensor = torch.from_numpy(lidar[:,:3])
        n_points,_= lidar_tensor.shape
        tran = tran.view(1,-1).repeat(n_points,1)
        rot = rot.repeat(n_points,1,1).type(torch.float32)
        # lidar_tensor = lidar_tensor - tran
        # lidar_depth = rot_inv.matmul(lidar_tensor)
        rotated_point_cloud = torch.einsum('ijk,ik->ij', rot, lidar_tensor)
        rotated_point_cloud += tran
        depths = rotated_point_cloud[..., 2:3]
        coloring = depths

        # # Take the actual picture (matrix multiplication with camera-matrix
        # # + renormalization).
        points = view_points(rotated_point_cloud,
                             intrins,
                             normalize=True)

        mask =torch.ones(n_points,dtype=torch.bool)
        mask &= (depths.squeeze(-1) > 0.0)
        mask &= (points[:,0] > 0)
        #TODO: W is 600?
        mask &= (points[:,0] < 800 - 1)
        mask &= (points[:,1] > 0)
        mask &= (points[:,1] < 600 - 1)

        points = points[mask]
        coloring = coloring[mask]

        #N,3
        pre_depth = torch.cat((points[:,:2],coloring),dim=-1)

        return pre_depth

    def get_lidar_depth_1(self, ext_matrix, intrins,lidar):

        n_points = lidar.shape[0]

        ext_matrix = np.linalg.inv(ext_matrix)[:3, :4]
        xyz_hom = np.concatenate(
            [lidar[:,:3], np.ones((lidar.shape[0], 1), dtype=np.float32)], axis=1)
        rotated_point_cloud = (ext_matrix @ xyz_hom.T).T
        rotated_point_cloud = torch.from_numpy(rotated_point_cloud)
        depths = rotated_point_cloud[..., 2:3]
        coloring = depths

        # # Take the actual picture (matrix multiplication with camera-matrix
        # # + renormalization).
        points = view_points(rotated_point_cloud,
                             intrins,
                             normalize=True)

        mask = torch.ones(n_points, dtype=torch.bool)
        mask &= (depths.squeeze(-1) > 0.0)
        mask &= (points[:, 0] > 0)
        # TODO: W is 600?
        mask &= (points[:, 0] < 600 - 1)
        mask &= (points[:, 1] > 0)
        mask &= (points[:, 1] < 800 - 1)

        points = points[mask]
        coloring = coloring[mask]

        # N,3
        pre_depth = torch.cat((points[:, :2], coloring), dim=-1)

        return pre_depth



        # from ego --> camera   only ego V
        # image_inputs = datadict['image_inputs']  # 4,3,320,480
        # lidar_np = datadict['lidar_np']
        # imgs = image_inputs['imgs']
        # intrins = image_inputs['intrins']  # 4 ,3 ,3
        # rots = image_inputs['rots']  # 4,3,3
        # trans = image_inputs['trans']  # 4,3

        # rots = torch.inverse(rots)
        # lidar = lidar_np[:,:3]
        # lidar_tensor = torch.from_numpy(lidar)
        # n_points, _ = lidar_tensor.shape
        # n_cam,_,H,W = imgs.shape
        # lidar_tensor = lidar_tensor.repeat(n_cam,1,1) #4,N,3
        # rots_points = rots.view(n_cam,1,3,3).repeat(1,n_points,1,1)
        # trans_points = trans.view(n_cam,1,-1).repeat(1,n_points,1)
        #
        # lidar_tensor = lidar_tensor-trans_points
        # # lidar_depth = rots_points.matmul(lidar_tensor)
        #
        # # 使用 Einstein summation 来进行批量点乘
        # rotated_point_cloud = torch.einsum('ijkl,ijl->ijk', rots_points, lidar_tensor)
        #
        # # 4, N ,1
        # depths = rotated_point_cloud[...,2:3]
        # coloring = depths
        # # Take the actual picture (matrix multiplication with camera-matrix
        # # + renormalization).
        # # points --> 4, N , 4
        # points = view_points(rotated_point_cloud,
        #                      intrins,
        #                      normalize=True)
        # mask =torch.ones(n_cam,n_points,dtype=torch.bool)
        # mask &= (depths.squeeze(-1) > 0.0)
        # mask &= (points[:,:,0] > 0)
        # mask &= (points[:,:,0] < W - 1)
        # mask &= (points[:,:,1] > 0)
        # mask &= (points[:,:,1] < H - 1)
        #
        # filtered_points_list = []  # 从你的过滤步骤中获取
        # color_points_list = []
        # for i in range(n_cam):
        #     filtered_points_list.append(points[i][mask[i].squeeze(-1)])
        #     color_points_list.append(coloring[i][mask[i].squeeze(-1)])
        # max_length = max([x.shape[0] for x in filtered_points_list])
        # padded_points = torch.full((4, max_length, 3), -1.0)
        # padded_color = torch.full((4, max_length, 1), -1.0)
        #
        # for i in range(4):
        #     length = filtered_points_list[i].shape[0]
        #     # print("Shape of filtered_points_list[i]:", filtered_points_list[i].shape)
        #     # print("Shape of padded_points[i, :length, :]:", padded_points[i, :length, :].shape)
        #     padded_points[i,:length,:] = filtered_points_list[i]
        #     padded_color[i,:length,:] = color_points_list[i]
        #
        # return padded_points,padded_color

    @staticmethod
    def merge_features_to_dict(processed_feature_list, merge=None):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)  # merged_feature_dict['coords'] = [f1,f2,f3,f4]

        # stack them
        # it usually happens when merging cavs images -> v.shape = [N, Ncam, C, H, W]
        # cat them
        # it usually happens when merging batches cav images -> v is a list [(N1+N2+...Nn, Ncam, C, H, W))]
        if merge == 'stack':
            for feature_name, features in merged_feature_dict.items():
                merged_feature_dict[feature_name] = torch.stack(features, dim=0)
        elif merge == 'cat':
            for feature_name, features in merged_feature_dict.items():
                merged_feature_dict[feature_name] = torch.cat(features, dim=0)

        return merged_feature_dict

    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        image_inputs_list = []
        # used to record different scenario
        record_len = []
        label_dict_list = []
        lidar_pose_list = []
        lidar_pose_clean_list = []

        # pairwise transformation matrix
        pairwise_t_matrix_list = []

        #
        lidar2img_matrix_list = []
        if self.visualize:
            origin_lidar = []

        ### 2022.10.10 single gt ####
        if self.supervise_single:
            pos_equal_one_single = []
            neg_equal_one_single = []
            targets_single = []
        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])
            lidar_pose_list.append(ego_dict['lidar_poses'])  # ego_dict['lidar_pose'] is np.ndarray [N,6]
            lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])

            image_inputs_list.append(ego_dict['image_inputs'])  # different cav_num, ego_dict['image_inputs'] is dict.
            record_len.append(ego_dict['cav_num'])

            label_dict_list.append(ego_dict['label_dict'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

            lidar2img_matrix_list.append(ego_dict['lidar2img_matrix'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

            ### 2022.10.10 single gt ####
            if self.supervise_single:
                pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                targets_single.append(ego_dict['single_label_dict_torch']['targets'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))
        lidar2img_matrix = torch.from_numpy(np.array(lidar2img_matrix_list))

        # {"image_inputs":
        #   {image: [sum(record_len), Ncam, C, H, W]}
        # }
        merged_image_inputs_dict = self.merge_features_to_dict(image_inputs_list, merge='cat')

        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        # [[N1, 6], [N2, 6]...] -> [[N1+N2+...], 6]
        lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
        lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)

        # (B, max_cav)
        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

        # add pairwise_t_matrix to label dict
        label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
        label_torch_dict['record_len'] = record_len
        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'image_inputs': merged_image_inputs_dict,
                                   'record_len': record_len,
                                   'label_dict': label_torch_dict,
                                   'object_ids': object_ids[0],
                                   'pairwise_t_matrix': pairwise_t_matrix,
                                   'lidar2img_matrix': lidar2img_matrix,
                                   'lidar_pose_clean': lidar_pose_clean,
                                    'lidar_3d':batch[i]['ego']['lidar_3d'],
                                   'lidar_pose': lidar_pose})

        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        ### 2022.10.10 single gt ####
        if self.supervise_single:
            output_dict['ego'].update({
                "label_dict_single":
                    {"pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                     "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                     "targets": torch.cat(targets_single, dim=0)}
            })

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)
        if output_dict is None:
            return None

        # check if anchor box in the batch
        output_dict['ego'].update({'anchor_box':
                                       self.anchor_box_torch})

        # save the transformation matrix (4, 4) to ego vehicle
        # transformation is only used in post process (no use.)
        # we all predict boxes in ego coord.
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        transformation_matrix_clean_torch = \
            torch.from_numpy(np.identity(4)).float()

        output_dict['ego'].update({'transformation_matrix':
                                       transformation_matrix_torch,
                                   'transformation_matrix_clean':
                                       transformation_matrix_clean_torch, })

        output_dict['ego'].update({
            "sample_idx": batch[0]['ego']['sample_idx'],
            "cav_id_list": batch[0]['ego']['cav_id_list']
        })

        return output_dict

    def generate_object_center(
            self, cav_contents, reference_lidar_pose, visibility_map
    ):
        """
        Retrieve all objects in a format of (n, 7), where 7 represents
        x, y, z, l, w, h, yaw or x, y, z, h, w, l, yaw.
        The object_bbx_center is in ego coordinate.

        Notice: it is a wrap of postprocessor

        Parameters
        ----------
        cav_contents : list
            List of dictionary, save all cavs' information.
            in fact it is used in get_item_single_car, so the list length is 1

        reference_lidar_pose : list
            The final target lidar pose with length 6.

        visibility_map : np.ndarray
            for OPV2V, its 256*256 resolution. 0.39m per pixel. heading up.

        Returns
        -------
        object_np : np.ndarray
            Shape is (max_num, 7).
        mask : np.ndarray
            Shape is (max_num,).
        object_ids : list
            Length is number of bbx in current sample.
        """
        return self.post_processor.generate_visible_object_center(
            cav_contents, reference_lidar_pose, visibility_map
        )

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4), L is the max cav number in a scene
            pairwise_t_matrix[i, j] is Tji, i_to_j
        """
        pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1))  # (L, L, 4, 4)

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            # no need to warp again in fusion time.

            # pairwise_t_matrix[:, :] = np.identity(4)
            return pairwise_t_matrix
        else:
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                lidar_pose = cav_content['params']['lidar_pose']
                t_list.append(x_to_world(lidar_pose))  # Twx

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i != j:
                        # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                        # t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                        t_matrix = np.linalg.solve(t_list[j], t_list[i])  # Tjw*Twi = Tji
                        pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix


def view_points(points, view, normalize: bool):
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """
    # view: 4,3,3
    # points: n_cam,N,3
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[1] == 3

    N,_ = points.shape
    viewpad = torch.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    points = torch.cat((points,torch.ones(N,1)),dim=-1)
    points = torch.einsum('ij, kj -> ki', viewpad, points)
    points = points[:,:3]

    if normalize:
        points = points / points[:,2:3].repeat(1,3).reshape(N,3)
    # viewpad = torch.eye(4).unsqueeze(0).repeat(n_cam,1,1)
    # viewpad[:, :view.shape[1], :view.shape[2]] = view
    #
    # nbr_points = points.shape[1]
    #
    # # Do operation in homogenous coordinates.
    # # 4, N , 4
    # pad = torch.ones(4, nbr_points, 1)
    # points = torch.cat((points,pad),dim=-1)
    # transformed_points = torch.einsum('ikl, ijk->ijl', viewpad, points)
    # # points = np.dot(viewpad, points)
    # points = transformed_points[:,:,:3]
    #
    # if normalize:
    #     points = points / points[:,:,2:3].repeat(1,1,3).reshape(-1,nbr_points,3)

    return points









