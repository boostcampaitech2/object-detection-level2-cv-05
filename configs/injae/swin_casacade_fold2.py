_base_ = [
    './base_models.py',
    './base_dataset.py',
    './base_schedules.py',
    './base_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))
    # neck=dict(in_channels=[256, 512, 1024, 2048]))



data = dict(
            samples_per_gpu=8,
            train=dict(
                ann_file='../dataset/split_train_2.json'
            ), 
            val=dict(
                ann_file='../dataset/split_valid_2.json'
            )
            )

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='CosineRestart',
    # policy='CosineAnnealing',
    periods=[25, 30],
    restart_weights=[1, 0.1],
    min_lr= 1e-5,
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001)



checkpoint_config = dict(
    max_keep_ckpts=3)

runner = dict(type='EpochBasedRunner', max_epochs=30)
