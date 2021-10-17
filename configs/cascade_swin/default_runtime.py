checkpoint_config = dict(max_keep_ckpts=2, interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        #dict(type='TextLoggerHook')
        dict(type='MyHook', early_stopping=36),
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

gpu_ids = [0]
seed = 2147

evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50', classwise=True)