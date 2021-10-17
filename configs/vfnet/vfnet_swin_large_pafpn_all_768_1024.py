_base_ = 'vfnet_swin_large_pafpn.py'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile',to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [dict(
                type='Resize',
                img_scale=[(704, 1333), (736, 1333), (768, 1333), (800, 1333),
                           (832, 1333), (864, 1333), (896, 1333), (928, 1333),
                           (960, 1333), (992, 1333), (1024,1333)],
                multiscale_mode='value',
                keep_ratio=True)],
            [dict(
                  type='Resize',
                  img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                  multiscale_mode='value',
                  keep_ratio=True),
              dict(
                  type='RandomCrop',
                  crop_type='absolute_range',
                  crop_size=(384, 600),
                  allow_negative_crop=True),
              dict(
                  type='Resize',
                  img_scale=[(704, 1333), (736, 1333), (768, 1333), (800, 1333),
                             (832, 1333), (864, 1333), (896, 1333), (928, 1333),
                             (960, 1333), (992, 1333), (1024,1333)],
                  multiscale_mode='value',
                  override=True,
                  keep_ratio=True)],
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    samples_per_gpu=2,
    train = dict(
        ann_file = '/opt/ml/detection/dataset/train_mosaic.json',
        pipeline = train_pipeline),
    val = None)
