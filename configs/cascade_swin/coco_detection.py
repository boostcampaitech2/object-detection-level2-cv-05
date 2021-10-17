# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale_ = (1333, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=img_scale_, keep_ratio=True),
    dict(type='CutOut',n_holes=10, cutout_ratio=[(0.1, 0.1), (0.05, 0.05)]),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(704, 1333), (736, 1333), (768, 1333),
                           (800, 1333), (832, 1333), (864, 1333), 
                           (896, 1333), (928, 1333), (960, 1333),
                           (992, 1333), (1024, 1333)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(600, 1333), (700, 1333), (800, 1333)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(576, 800),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(704, 1333), (736, 1333), (768, 1333),
                                     (800, 1333), (832, 1333), (864, 1333), 
                                     (896, 1333), (928, 1333), (960, 1333),
                                     (992, 1333), (1024, 1333)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale_,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes=["General trash",
                "Paper",
                "Paper pack",
                "Metal",
                "Glass",
                "Plastic",
                "Styrofoam",
                "Plastic bag",
                "Battery",
                "Clothing"]),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'predicted_test.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=["General trash",
                "Paper",
                "Paper pack",
                "Metal",
                "Glass",
                "Plastic",
                "Styrofoam",
                "Plastic bag",
                "Battery",
                "Clothing"]),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=["General trash",
                "Paper",
                "Paper pack",
                "Metal",
                "Glass",
                "Plastic",
                "Styrofoam",
                "Plastic bag",
                "Battery",
                "Clothing"]))