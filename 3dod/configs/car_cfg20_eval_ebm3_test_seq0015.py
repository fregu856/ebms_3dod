model = dict(
    type='SingleStageDetector20',
    backbone=dict(
        type='SimpleVoxel',
        num_input_features=4,
        use_norm=True,
        num_filters=[32, 64],
        with_distance=False),

    neck=dict(
        type='SpMiddleFHD',
        output_shape=[40, 1600, 1408],
        num_input_features=4,
        num_hidden_features=64 * 5,),

    bbox_head=dict(
        type='SSDRotateHead',
        num_class=1,
        num_output_filters=256,
        num_anchor_per_loc=2,
        use_sigmoid_cls=True,
        encode_rad_error_by_sin=True,
        use_direction_classifier=True,
        box_code_size=7,),

    extra_head=dict(
        type='PSWarpHead',
        grid_offsets = (0., 40.),
        featmap_stride=.4,
        in_channels=256,
        num_class=1,
        num_parts=28,)
)

# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45, # this one is to limit the force assignment
            ignore_iof_thr=-1,
            similarity_fn ='NearestIouSimilarity'
        ),
        nms=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            nms_thr=0.7,
            min_bbox_size=0
        ),
        allowed_border=0,
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,
        debug=False),
    extra=dict(
        assigner=dict(
            pos_iou_thr=0.7,
            neg_iou_thr=0.7,
            min_pos_iou=0.7,
            ignore_iof_thr=-1,
            similarity_fn ='RotateIou3dSimilarity'
        )
    )
)

test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=100,
        nms_thr=0.7,
        min_bbox_size=0
    ),
    extra=dict(
        score_thr=0.3, nms=dict(type='nms', iou_thr=0.1), max_per_img=100, EBM_guided=False, EBM_refine=True, EBM_refine_steps=10)
)
# # dataset settings
# dataset_type = 'KittiLiDAR'
# data_root = '/root/ebms_3dod/3dod/data/KITTI/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# data = dict(
#     imgs_per_gpu=2,
#     # workers_per_gpu=4,
#     workers_per_gpu=1,
#     train=dict(
#         type=dataset_type,
#         root=data_root + 'object/training/',
#         ann_file=data_root + 'ImageSets/train.txt',
#         img_prefix=None,
#         img_scale=(1242, 375),
#         img_norm_cfg=img_norm_cfg,
#         size_divisor=32,
#         flip_ratio=0.5,
#         with_mask=False,
#         with_label=True,
#         with_point=True,
#         class_names = ['Car', 'Van'],
#         augmentor=dict(
#             type='PointAugmentor',
#             root_path=data_root,
#             info_path=data_root + 'kitti_dbinfos_trainval.pkl',
#             sample_classes=['Car'],
#             min_num_points=5,
#             sample_max_num=15,
#             removed_difficulties=[-1],
#             global_rot_range=[-0.78539816, 0.78539816],
#             gt_rot_range=[-0.78539816, 0.78539816],
#             center_noise_std=[1., 1., .5],
#             scale_range=[0.95, 1.05]
#         ),
#         generator=dict(
#             type='VoxelGenerator',
#             voxel_size=[0.05, 0.05, 0.1],
#             point_cloud_range=[0, -40., -3., 70.4, 40., 1.],
#             max_num_points=5,
#             max_voxels=20000
#         ),
#         anchor_generator=dict(
#             type='AnchorGeneratorStride',
#             sizes=[1.6, 3.9, 1.56],
#             anchor_strides=[0.4, 0.4, 1.0],
#             anchor_offsets=[0.2, -39.8, -1.78],
#             rotations=[0, 1.57],
#         ),
#         anchor_area_threshold=1,
#         out_size_factor=8,
#         test_mode=False),
#
#     val=dict(
#         type=dataset_type,
#         root=data_root + 'object/testing/',
#         ann_file=data_root + 'ImageSets/test.txt',
#         img_prefix=None,
#         img_scale=(1242, 375),
#         img_norm_cfg=img_norm_cfg,
#         size_divisor=32,
#         flip_ratio=0,
#         with_mask=False,
#         with_label=False,
#         with_point=True,
#         class_names = ['Car'],
#         generator=dict(
#             type='VoxelGenerator',
#             voxel_size=[0.05, 0.05, 0.1],
#             point_cloud_range=[0., -40., -3., 70.4, 40., 1.],
#             max_num_points=5,
#             max_voxels=20000
#         ),
#         anchor_generator=dict(
#             type='AnchorGeneratorStride',
#             sizes=[1.6, 3.9, 1.56],
#             anchor_strides=[0.4, 0.4, 1.0],
#             anchor_offsets=[0.2, -39.8, -1.78],
#             rotations=[0, 1.57],
#         ),
#         anchor_area_threshold=1,
#         out_size_factor=8,
#         test_mode=True),
# )
# dataset settings
dataset_type = 'KittiVideo'
data_root = '/root/ebms_3dod/3dod/data/KITTI/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    # workers_per_gpu=4,
    workers_per_gpu=1,
    val=dict(
        type=dataset_type,
        root=data_root + 'tracking/testing/',
        calib_dir = 'calib/0015.txt',
        img_dir = 'image_02/0015',
        lidar_dir = 'velodyne/0015',
        ann_file=data_root + 'ImageSets/test.txt',
        img_prefix=None,
        img_scale=(1242, 375),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        with_point=True,
        class_names = ['Car'],
        generator=dict(
            type='VoxelGenerator',
            voxel_size=[0.05, 0.05, 0.1],
            point_cloud_range=[0., -40., -3., 70.4, 40., 1.],
            max_num_points=5,
            max_voxels=20000
        ),
        anchor_generator=dict(
            type='AnchorGeneratorStride',
            sizes=[1.6, 3.9, 1.56],
            anchor_strides=[0.4, 0.4, 1.0],
            anchor_offsets=[0.2, -39.8, -1.78],
            rotations=[0, 1.57],
        ),
        anchor_area_threshold=1,
        out_size_factor=8,
        test_mode=True),
)
# optimizer
optimizer = dict(
    type='adam_onecycle', lr=0.003, weight_decay=0.01,
    grad_clip=dict(max_norm=10, norm_type=2)
)
# learning policy
lr_config = dict(
    policy='onecycle',
    moms = [0.95, 0.85],
    div_factor = 10,
    pct_start = 0.4
)

checkpoint_config = dict(interval=5)
log_config = dict(interval=50)

total_epochs = 80
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = '../saved_model_vehicle
work_dir = '/root/ebms_3dod/3dod/saved_model_vehicle20'
load_from = None
resume_from = None
workflow = [('train', 1)]
SA_SSD_pretrained = True
SA_SSD_fixed = True
USE_EBM = True
