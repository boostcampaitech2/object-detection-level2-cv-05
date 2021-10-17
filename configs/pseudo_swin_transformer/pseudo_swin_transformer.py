_base_ = [
    './base_models.py',
    './base_dataset.py',
    './base_runtime.py'
]
dataset_type = 'PseudoCocoDataset'

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

resume_from = '/opt/ml/detection/mmdetection/work_dirs/swin_casacade_fold2/best_bbox_mAP_50_epoch_24.pth'
# resume_from = '/opt/ml/detection/mmdetection/work_dirs/pseudo_label/swin_transformer/epoch_41.pth'

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
                type="PseudoCocoDataset",
                ann_file='../dataset/' + 'pseudo_train.json'
            )
)



optimizer = dict(
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
optimizer_config = dict(grad_clip=None)


lr_config = dict(
    policy='step',
    # warmup='linear',
    # warmup_iters=500,
    # warmup_ratio=0.001,
    step=[24+6, 24+10])
runner = dict(type='EpochBasedRunner', max_epochs=24+14)




checkpoint_config = dict(
    max_keep_ckpts=1)