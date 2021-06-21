import torch
import torch.utils.data

import os
import numpy as np
import pickle
import cv2

from utils import LabelLoader2D3D, calibread, wrapToPi

import open3d
import math

def draw_geometries_dark_background(geometries):
    vis = open3d.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.run()
    vis.destroy_window()

def draw_geometries_light_background(geometries):
    vis = open3d.Visualizer()
    vis.create_window()
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.run()
    vis.destroy_window()

def create3Dbbox(center, h, w, l, r_y, type="pred"):
    if type == "pred":
        color = [1, 0.75, 0] # (normalized RGB)
        front_color = [1, 0, 0] # (normalized RGB)
    else: # (if type == "gt":)
        color = [1, 0, 0.75] # (normalized RGB)
        front_color = [0, 0.9, 1] # (normalized RGB)

    Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)],
                       [0, 1, 0],
                       [-math.sin(r_y), 0, math.cos(r_y)]],
                       dtype='float32')

    Rmat_90 = np.asarray([[math.cos(r_y+np.pi/2), 0, math.sin(r_y+np.pi/2)],
                          [0, 1, 0],
                          [-math.sin(r_y+np.pi/2), 0, math.cos(r_y+np.pi/2)]],
                          dtype='float32')

    Rmat_90_x = np.asarray([[1, 0, 0],
                            [0, math.cos(np.pi/2), math.sin(np.pi/2)],
                            [0, -math.sin(np.pi/2), math.cos(np.pi/2)]],
                            dtype='float32')

    p0 = center + np.dot(Rmat, np.asarray([l/2.0, 0, w/2.0], dtype='float32').flatten())
    p1 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, w/2.0], dtype='float32').flatten())
    p2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p4 = center + np.dot(Rmat, np.asarray([l/2.0, -h, w/2.0], dtype='float32').flatten())
    p5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, w/2.0], dtype='float32').flatten())
    p6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, -w/2.0], dtype='float32').flatten())
    p7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, -w/2.0], dtype='float32').flatten())

    p0_3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, 0], dtype='float32').flatten())
    p1_2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, 0], dtype='float32').flatten())
    p4_7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, 0], dtype='float32').flatten())
    p5_6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, 0], dtype='float32').flatten())
    p0_1 = center + np.dot(Rmat, np.asarray([0, 0, w/2.0], dtype='float32').flatten())
    p3_2 = center + np.dot(Rmat, np.asarray([0, 0, -w/2.0], dtype='float32').flatten())
    p4_5 = center + np.dot(Rmat, np.asarray([0, -h, w/2.0], dtype='float32').flatten())
    p7_6 = center + np.dot(Rmat, np.asarray([0, -h, -w/2.0], dtype='float32').flatten())
    p0_4 = center + np.dot(Rmat, np.asarray([l/2.0, -h/2.0, w/2.0], dtype='float32').flatten())
    p3_7 = center + np.dot(Rmat, np.asarray([l/2.0, -h/2.0, -w/2.0], dtype='float32').flatten())
    p1_5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h/2.0, w/2.0], dtype='float32').flatten())
    p2_6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h/2.0, -w/2.0], dtype='float32').flatten())
    p0_1_3_2 = center

    length_0_3 = np.linalg.norm(p0 - p3)
    cylinder_0_3 = open3d.create_mesh_cylinder(radius=0.025, height=length_0_3)
    cylinder_0_3.compute_vertex_normals()
    transform_0_3 = np.eye(4)
    transform_0_3[0:3, 0:3] = Rmat
    transform_0_3[0:3, 3] = p0_3
    cylinder_0_3.transform(transform_0_3)
    cylinder_0_3.paint_uniform_color(front_color)

    length_1_2 = np.linalg.norm(p1 - p2)
    cylinder_1_2 = open3d.create_mesh_cylinder(radius=0.025, height=length_1_2)
    cylinder_1_2.compute_vertex_normals()
    transform_1_2 = np.eye(4)
    transform_1_2[0:3, 0:3] = Rmat
    transform_1_2[0:3, 3] = p1_2
    cylinder_1_2.transform(transform_1_2)
    cylinder_1_2.paint_uniform_color(color)

    length_4_7 = np.linalg.norm(p4 - p7)
    cylinder_4_7 = open3d.create_mesh_cylinder(radius=0.025, height=length_4_7)
    cylinder_4_7.compute_vertex_normals()
    transform_4_7 = np.eye(4)
    transform_4_7[0:3, 0:3] = Rmat
    transform_4_7[0:3, 3] = p4_7
    cylinder_4_7.transform(transform_4_7)
    cylinder_4_7.paint_uniform_color(front_color)

    length_5_6 = np.linalg.norm(p5 - p6)
    cylinder_5_6 = open3d.create_mesh_cylinder(radius=0.025, height=length_5_6)
    cylinder_5_6.compute_vertex_normals()
    transform_5_6 = np.eye(4)
    transform_5_6[0:3, 0:3] = Rmat
    transform_5_6[0:3, 3] = p5_6
    cylinder_5_6.transform(transform_5_6)
    cylinder_5_6.paint_uniform_color(color)

    # #

    length_0_1 = np.linalg.norm(p0 - p1)
    cylinder_0_1 = open3d.create_mesh_cylinder(radius=0.025, height=length_0_1)
    cylinder_0_1.compute_vertex_normals()
    transform_0_1 = np.eye(4)
    transform_0_1[0:3, 0:3] = Rmat_90
    transform_0_1[0:3, 3] = p0_1
    cylinder_0_1.transform(transform_0_1)
    cylinder_0_1.paint_uniform_color(color)

    length_3_2 = np.linalg.norm(p3 - p2)
    cylinder_3_2 = open3d.create_mesh_cylinder(radius=0.025, height=length_3_2)
    cylinder_3_2.compute_vertex_normals()
    transform_3_2 = np.eye(4)
    transform_3_2[0:3, 0:3] = Rmat_90
    transform_3_2[0:3, 3] = p3_2
    cylinder_3_2.transform(transform_3_2)
    cylinder_3_2.paint_uniform_color(color)

    length_4_5 = np.linalg.norm(p4 - p5)
    cylinder_4_5 = open3d.create_mesh_cylinder(radius=0.025, height=length_4_5)
    cylinder_4_5.compute_vertex_normals()
    transform_4_5 = np.eye(4)
    transform_4_5[0:3, 0:3] = Rmat_90
    transform_4_5[0:3, 3] = p4_5
    cylinder_4_5.transform(transform_4_5)
    cylinder_4_5.paint_uniform_color(color)

    length_7_6 = np.linalg.norm(p7 - p6)
    cylinder_7_6 = open3d.create_mesh_cylinder(radius=0.025, height=length_7_6)
    cylinder_7_6.compute_vertex_normals()
    transform_7_6 = np.eye(4)
    transform_7_6[0:3, 0:3] = Rmat_90
    transform_7_6[0:3, 3] = p7_6
    cylinder_7_6.transform(transform_7_6)
    cylinder_7_6.paint_uniform_color(color)

    # #

    length_0_4 = np.linalg.norm(p0 - p4)
    cylinder_0_4 = open3d.create_mesh_cylinder(radius=0.025, height=length_0_4)
    cylinder_0_4.compute_vertex_normals()
    transform_0_4 = np.eye(4)
    transform_0_4[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_0_4[0:3, 3] = p0_4
    cylinder_0_4.transform(transform_0_4)
    cylinder_0_4.paint_uniform_color(front_color)

    length_3_7 = np.linalg.norm(p3 - p7)
    cylinder_3_7 = open3d.create_mesh_cylinder(radius=0.025, height=length_3_7)
    cylinder_3_7.compute_vertex_normals()
    transform_3_7 = np.eye(4)
    transform_3_7[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_3_7[0:3, 3] = p3_7
    cylinder_3_7.transform(transform_3_7)
    cylinder_3_7.paint_uniform_color(front_color)

    length_1_5 = np.linalg.norm(p1 - p5)
    cylinder_1_5 = open3d.create_mesh_cylinder(radius=0.025, height=length_1_5)
    cylinder_1_5.compute_vertex_normals()
    transform_1_5 = np.eye(4)
    transform_1_5[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_1_5[0:3, 3] = p1_5
    cylinder_1_5.transform(transform_1_5)
    cylinder_1_5.paint_uniform_color(color)

    length_2_6 = np.linalg.norm(p2 - p6)
    cylinder_2_6 = open3d.create_mesh_cylinder(radius=0.025, height=length_2_6)
    cylinder_2_6.compute_vertex_normals()
    transform_2_6 = np.eye(4)
    transform_2_6[0:3, 0:3] = np.dot(Rmat, Rmat_90_x)
    transform_2_6[0:3, 3] = p2_6
    cylinder_2_6.transform(transform_2_6)
    cylinder_2_6.paint_uniform_color(color)

    # #

    length_0_1_3_2 = np.linalg.norm(p0_1 - p3_2)
    cylinder_0_1_3_2 = open3d.create_mesh_cylinder(radius=0.025, height=length_0_1_3_2)
    cylinder_0_1_3_2.compute_vertex_normals()
    transform_0_1_3_2 = np.eye(4)
    transform_0_1_3_2[0:3, 0:3] = Rmat
    transform_0_1_3_2[0:3, 3] = p0_1_3_2
    cylinder_0_1_3_2.transform(transform_0_1_3_2)
    cylinder_0_1_3_2.paint_uniform_color(color)

    return [cylinder_0_1_3_2, cylinder_0_3, cylinder_1_2, cylinder_4_7, cylinder_5_6, cylinder_0_1, cylinder_3_2, cylinder_4_5, cylinder_7_6, cylinder_0_4, cylinder_3_7, cylinder_1_5, cylinder_2_6]

def create3Dbbox_poly(center, h, w, l, r_y, P2_mat, type="pred"):
    if type == "pred":
        color = [0, 190, 255] # (BGR)
        front_color = [0, 0, 255] # (BGR)
    else: # (if type == "gt":)
        color = [190, 0, 255] # (BGR)
        front_color = [255, 230, 0] # (BGR)

    poly = {}

    Rmat = np.asarray([[math.cos(r_y), 0, math.sin(r_y)],
                       [0, 1, 0],
                       [-math.sin(r_y), 0, math.cos(r_y)]],
                       dtype='float32')

    p0 = center + np.dot(Rmat, np.asarray([l/2.0, 0, w/2.0], dtype='float32').flatten())
    p1 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, w/2.0], dtype='float32').flatten())
    p2 = center + np.dot(Rmat, np.asarray([-l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p3 = center + np.dot(Rmat, np.asarray([l/2.0, 0, -w/2.0], dtype='float32').flatten())
    p4 = center + np.dot(Rmat, np.asarray([l/2.0, -h, w/2.0], dtype='float32').flatten())
    p5 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, w/2.0], dtype='float32').flatten())
    p6 = center + np.dot(Rmat, np.asarray([-l/2.0, -h, -w/2.0], dtype='float32').flatten())
    p7 = center + np.dot(Rmat, np.asarray([l/2.0, -h, -w/2.0], dtype='float32').flatten())

    poly['points'] = np.array([p0, p1, p2, p3, p4, p5, p6, p7])
    poly['lines'] = [[0, 3, 7, 4, 0], [1, 2, 6, 5, 1], [0, 1], [2, 3], [6, 7], [4, 5]] # (0 -> 3 -> 7 -> 4 -> 0, 1 -> 2 -> 6 -> 5 -> 1, etc.)
    poly['colors'] = [front_color, color, color, color, color, color]
    poly['P0_mat'] = P2_mat

    return poly

def draw_3d_polys(img, polys):
    img = np.copy(img)
    for poly in polys:
        for n, line in enumerate(poly['lines']):
            if 'colors' in poly:
                bg = poly['colors'][n]
            else:
                bg = np.array([255, 0, 0], dtype='float64')

            p3d = np.vstack((poly['points'][line].T, np.ones((1, poly['points'][line].shape[0]))))
            p2d = np.dot(poly['P0_mat'], p3d)

            for m, p in enumerate(p2d[2, :]):
                p2d[:, m] = p2d[:, m]/p

            cv2.polylines(img, np.int32([p2d[:2, :].T]), False, bg, lineType=cv2.LINE_AA, thickness=2)

    return img


for sequence in ["0011"]:
    print (sequence)

    project_dir = "/home/fredrik/ebms_3dod/3dod/" # NOTE! you'll have to adapt this for your file structure
    data_dir = project_dir + "data/KITTI/tracking/testing/"
    img_dir = data_dir + "image_02/" + sequence + "/"
    calib_path = project_dir + "data/KITTI/tracking/testing/calib/" + sequence + ".txt"
    lidar_dir = data_dir + "velodyne/" + sequence + "/"

    calib = calibread(calib_path)
    P2 = calib["P2"]
    Tr_velo_to_cam_orig = calib["Tr_velo_to_cam"]
    R0_rect_orig = calib["R0_rect"]

    R0_rect = np.eye(4)
    R0_rect[0:3, 0:3] = R0_rect_orig

    Tr_velo_to_cam = np.eye(4)
    Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig

    sorted_img_ids = []
    img_names = sorted(os.listdir(img_dir))
    for img_name in img_names:
        img_id = img_name.split(".png")[0]
        sorted_img_ids.append(img_id)

    img_height = 375
    img_width = 1242

    small_img_height = 225
    small_img_width = 744

    # ################################################################################
    # # create a video of images (no bboxes):
    # ################################################################################
    # # out = cv2.VideoWriter("eval_test_seq_%s_img.avi" % sequence, cv2.VideoWriter_fourcc(*'H264'), 12, (img_width, img_height), True)
    # out = cv2.VideoWriter("eval_test_seq_%s_img.mp4" % sequence, cv2.VideoWriter_fourcc(*"mp4v"), 12, (img_width, img_height), True)
    #
    # for img_id in sorted_img_ids:
    #     print (img_id)
    #
    #     img = cv2.imread(img_dir + img_id + ".png", -1)
    #
    #     img = cv2.resize(img, (img_width, img_height)) # (the image MUST have the size specified in VideoWriter)
    #
    #     cv2.imwrite("test.png", img)
    #
    #     out.write(img)

    class ImgCreatorLiDAR:
        def __init__(self):
            self.counter = 0
            self.trajectory = open3d.read_pinhole_camera_trajectory("/home/fredrik/ebms_3dod/3dod/camera_trajectory.json") # NOTE! you'll have to adapt this for your file structure

        def move_forward(self, vis):
            # this function is called within the Visualizer::run() loop.
            # the run loop calls the function, then re-renders the image.

            if self.counter < 2: # (the counter is for making sure the camera view has been changed before the img is captured)
                # set the camera view:
                ctr = vis.get_view_control()
                ctr.convert_from_pinhole_camera_parameters(self.trajectory.parameters[0])

                self.counter += 1
            else:
                # capture an image:
                img = vis.capture_screen_float_buffer()
                img = 255*np.asarray(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.uint8) # (shape: (1025, 1853, 3))

                img_ext = np.zeros((1080, 1920, 3)).astype(np.uint8)
                img_ext[0:1025, (1920/2 - int(1853/2)):(1920/2 + int(1853/2)+1), :] = img

                self.lidar_img = img_ext

                # close the window:
                vis.destroy_window()

                self.counter = 0

            return False

        def create_img(self, geometries):
            vis = open3d.Visualizer()
            vis.create_window(width=1853, height=1025) # NOTE! NOTE! NOTE! NOTE!
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            for geometry in geometries:
                vis.add_geometry(geometry)
            vis.register_animation_callback(self.move_forward)
            vis.run()

            return self.lidar_img

    # ################################################################################
    # # create a video of LiDAR (no bboxes):
    # ################################################################################
    # # out_lidar = cv2.VideoWriter("eval_test_seq_%s_lidar.avi" % sequence, cv2.VideoWriter_fourcc(*'H264'), 12, (1920, 1080), True)
    # out_lidar = cv2.VideoWriter("eval_test_seq_%s_lidar.mp4" % sequence, cv2.VideoWriter_fourcc(*"mp4v"), 12, (1920, 1080), True)
    #
    # lidar_img_creator = ImgCreatorLiDAR()
    # for img_id in sorted_img_ids:
    #     print (img_id)
    #
    #     lidar_path = lidar_dir + img_id + ".bin"
    #     point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    #
    #     # remove points that are located behind the camera:
    #     point_cloud = point_cloud[point_cloud[:, 0] > -2.5, :]
    #
    #     point_cloud_xyz = point_cloud[:, 0:3]
    #     point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
    #     point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))
    #
    #     # transform the points into (rectified) camera coordinates:
    #     point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
    #     # normalize:
    #     point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
    #     point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
    #     point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
    #     point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]
    #
    #     pcd = open3d.PointCloud()
    #     pcd.points = open3d.Vector3dVector(point_cloud_xyz_camera)
    #     pcd.paint_uniform_color([0.65, 0.65, 0.65])
    #
    #     img = lidar_img_creator.create_img([pcd])
    #     out_lidar.write(img)
    #
    #     if img_id == "000050":
    #         break

    # ################################################################################
    # # create a video of LiDAR with pred:
    # ################################################################################
    # # out_lidar_pred = cv2.VideoWriter("eval_test_seq_%s_lidar_pred.avi" % sequence, cv2.VideoWriter_fourcc(*'H264'), 12, (1920, 1080), True)
    # out_lidar_pred = cv2.VideoWriter("eval_test_seq_%s_lidar_pred.mp4" % sequence, cv2.VideoWriter_fourcc(*"mp4v"), 12, (1920, 1080), True)
    #
    # lidar_img_creator = ImgCreatorLiDAR()
    # for img_id in sorted_img_ids:
    #     print (img_id)
    #
    #     lidar_path = lidar_dir + img_id + ".bin"
    #     point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    #
    #     # remove points that are located behind the camera:
    #     point_cloud = point_cloud[point_cloud[:, 0] > -2.5, :]
    #
    #     point_cloud_xyz = point_cloud[:, 0:3]
    #     point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
    #     point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))
    #
    #     # transform the points into (rectified) camera coordinates:
    #     point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
    #     # normalize:
    #     point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
    #     point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
    #     point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
    #     point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]
    #
    #     pcd = open3d.PointCloud()
    #     pcd.points = open3d.Vector3dVector(point_cloud_xyz_camera)
    #     pcd.paint_uniform_color([0.65, 0.65, 0.65])
    #
    #     #####################################
    #     if img_id == "000000":
    #         img_id = "0"
    #     else:
    #         img_id = img_id.lstrip("0")
    #
    #     with open("/home/fredrik/ebms_3dod/3dod/preds%s/%s.pkl" % (sequence, img_id), "rb") as file: # NOTE! you'll have to adapt this for your file structure
    #         bboxes_numpy = pickle.load(file) # (shape: (num_detections, 7))
    #
    #     bboxes_list = []
    #     for i in range(bboxes_numpy.shape[0]):
    #         bboxes_list += [bboxes_numpy[i]]
    #
    #     pred_bboxes = []
    #     for bbox in bboxes_list:
    #         pred_bbox = create3Dbbox(np.array([-bbox[1], -bbox[2], bbox[0]]), bbox[5], bbox[3], bbox[4], bbox[6], type="pred")
    #         pred_bboxes += pred_bbox
    #     #####################################
    #
    #     img = lidar_img_creator.create_img(pred_bboxes + [pcd])
    #     out_lidar_pred.write(img)
    #
    #     if img_id == "50":
    #         break

    # ################################################################################
    # # create a video of image and LiDAR (no bboxes):
    # ################################################################################
    # # out_lidar_img = cv2.VideoWriter("eval_test_seq_%s_lidar_img.avi" % sequence, cv2.VideoWriter_fourcc(*'H264'), 12, (1920, 1080), True)
    # out_lidar_img = cv2.VideoWriter("eval_test_seq_%s_lidar_img.mp4" % sequence, cv2.VideoWriter_fourcc(*"mp4v"), 12, (1920, 1080), True)
    #
    # lidar_img_creator = ImgCreatorLiDAR()
    # for img_id in sorted_img_ids:
    #     print (img_id)
    #
    #     img = cv2.imread(img_dir + img_id + ".png", -1)
    #     small_img = cv2.resize(img, (small_img_width, small_img_height))
    #
    #     lidar_path = lidar_dir + img_id + ".bin"
    #     point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    #
    #     # remove points that are located behind the camera:
    #     point_cloud = point_cloud[point_cloud[:, 0] > -2.5, :]
    #
    #     point_cloud_xyz = point_cloud[:, 0:3]
    #     point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
    #     point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))
    #
    #     # transform the points into (rectified) camera coordinates:
    #     point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
    #     # normalize:
    #     point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
    #     point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
    #     point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
    #     point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]
    #
    #     pcd = open3d.PointCloud()
    #     pcd.points = open3d.Vector3dVector(point_cloud_xyz_camera)
    #     pcd.paint_uniform_color([0.65, 0.65, 0.65])
    #
    #     img_lidar = lidar_img_creator.create_img([pcd])
    #
    #     combined_img = img_lidar
    #     combined_img[-small_img_height:, ((1920/2)-(small_img_width/2)):((1920/2)+(small_img_width/2))] = small_img
    #
    #     out_lidar_img.write(combined_img)
    #
    #     if img_id == "000050":
    #         break

    ################################################################################
    # create a video of image and LiDAR with pred:
    ################################################################################
    # out_lidar_img_pred = cv2.VideoWriter("eval_test_seq_%s_lidar_img_pred.avi" % sequence, cv2.VideoWriter_fourcc(*'H264'), 12, (1920, 1080), True)
    out_lidar_img_pred = cv2.VideoWriter("eval_test_seq_%s_lidar_img_pred.mp4" % sequence, cv2.VideoWriter_fourcc(*"mp4v"), 12, (1920, 1080), True)

    lidar_img_creator = ImgCreatorLiDAR()
    for img_id in sorted_img_ids:
        print (img_id)

        lidar_path = lidar_dir + img_id + ".bin"
        point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

        # remove points that are located behind the camera:
        point_cloud = point_cloud[point_cloud[:, 0] > -2.5, :]

        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3] # (point_cloud_xyz_hom has shape (num_points, 4))

        # transform the points into (rectified) camera coordinates:
        point_cloud_xyz_camera_hom = np.dot(R0_rect, np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T # (point_cloud_xyz_hom.T has shape (4, num_points))
        # normalize:
        point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
        point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1]/point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2]/point_cloud_xyz_camera_hom[:, 3]

        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(point_cloud_xyz_camera)
        pcd.paint_uniform_color([0.65, 0.65, 0.65])

        #####################################
        img = cv2.imread(img_dir + img_id + ".png", -1)

        if img_id == "000000":
            img_id = "0"
        else:
            img_id = img_id.lstrip("0")

        with open("/home/fredrik/ebms_3dod/3dod/preds%s/%s.pkl" % (sequence, img_id), "rb") as file: # NOTE! you'll have to adapt this for your file structure
            bboxes_numpy = pickle.load(file) # (shape: (num_detections, 7))

        bboxes_list_img_viz = []
        for i in range(bboxes_numpy.shape[0]):
            if bboxes_numpy[i][0] > 2.75:
                bboxes_list_img_viz += [bboxes_numpy[i]]

        pred_bbox_polys = []
        for bbox in bboxes_list_img_viz:
            pred_bbox_poly = create3Dbbox_poly(np.array([-bbox[1], -bbox[2], bbox[0]]), bbox[5], bbox[3], bbox[4], bbox[6], P2, type="pred")
            pred_bbox_polys.append(pred_bbox_poly)
        img_with_pred_bboxes = draw_3d_polys(img, pred_bbox_polys)
        small_img_with_pred_bboxes = cv2.resize(img_with_pred_bboxes, (small_img_width, small_img_height))

        bboxes_list_lidar_viz = []
        for i in range(bboxes_numpy.shape[0]):
            if bboxes_numpy[i][0] > 1.0:
                bboxes_list_lidar_viz += [bboxes_numpy[i]]

        pred_bboxes = []
        for bbox in bboxes_list_lidar_viz:
            pred_bbox = create3Dbbox(np.array([-bbox[1], -bbox[2], bbox[0]]), bbox[5], bbox[3], bbox[4], bbox[6], type="pred")
            pred_bboxes += pred_bbox
        #####################################

        img_lidar = lidar_img_creator.create_img(pred_bboxes + [pcd])

        combined_img = img_lidar
        combined_img[-small_img_height:, ((1920/2)-(small_img_width/2)):((1920/2)+(small_img_width/2))] = small_img_with_pred_bboxes

        out_lidar_img_pred.write(combined_img)

        # if img_id == "50":
        #     break
