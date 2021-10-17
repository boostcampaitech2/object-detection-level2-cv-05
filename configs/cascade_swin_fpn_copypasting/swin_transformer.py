_base_ = [
    './base_models.py',
    './base_dataset.py',
    './base_schedules.py',
    './default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        # arch='large',
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        # ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        attn_drop_rate=0.,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]))


data = dict(
            samples_per_gpu=8
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

lr_config = dict(warmup_iters=2000, step=[16, 22])


checkpoint_config = dict(
    max_keep_ckpts=3)