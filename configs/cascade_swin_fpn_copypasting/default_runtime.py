checkpoint_config = dict(max_keep_ckpts=3, interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        # dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(type='MyHook', early_stopping=7)
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

gpu_ids = [0]
seed = 2021