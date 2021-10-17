# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip= dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2400,
    warmup_ratio=0.001,
    step=[24, 33],
    gamma=0.1
    )
runner = dict(type='EpochBasedRunner', max_epochs=36)