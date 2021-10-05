_base_ = "faster_rcnn_r50_fpn_1x_coco.py"

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomCrop',
        crop_type='relative_range',
        crop_size=(0.75, 0.75),
        crop_p=0.5),
    dict(
        type='Resize',
        img_scale= (512,512),
        keep_ratio=True),
    dict(
        type='CutOut',
        n_holes=(5, 10),
        cutout_shape=[(4, 4), (4, 8), (8, 4), (8, 8),
                      (16, 8), (8, 16), (16, 16), (16, 32), (32, 16), (32, 32),
                      (32, 48), (48, 32), (48, 48)],
        cutout_p=0.5),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
