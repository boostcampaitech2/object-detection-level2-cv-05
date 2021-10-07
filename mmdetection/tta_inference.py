import numpy as np
import pandas as pd
import os
import argparse

import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from pandas import DataFrame
from pycocotools.coco import COCO

def main(config, scales, img_norm_cfg, weight, work_dir, samples_per_gpu, json_file='test.json', flip=True):
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # config file 들고오기
    cfg = Config.fromfile(config)

    root='/opt/ml/detection/dataset/'

    checkpoint_path = weight

    # dataset config 수정
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + json_file # default 'test.json'
    cfg.data.test.test_mode = True

    cfg.seed=1997
    cfg.work_dir = work_dir

    cfg.model.train_cfg = None

    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, f'{checkpoint_path}.pth')

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu') # ckpt load

    model.CLASSES = classes
    model = MMDataParallel(model.cuda(), device_ids=[0])

    if flip:
        flip_ratios=[0.0, 1.0]
    else:
        flip_ratios=[0.0]
    
    for scale in scales:
        for f_r in flip_ratios:
            test_pipeline = [
                dict(type='LoadImageFromFile'),
                dict(type='RandomFlip', flip_ratio=f_r),
                dict(type='MultiScaleFlipAug', img_scale=[scale], flip=False, 
                    transforms=[dict(type='Resize', keep_ratio=True),
                                dict(type='RandomFlip'),
                                dict(type='Normalize', **img_norm_cfg),
                                dict(type='Pad', size_divisor=32),
                                dict(type='ImageToTensor', keys=['img']),
                                dict(type='Collect', keys=['img'])]
                    )
            ]
            cfg.data.test.pipeline = test_pipeline

            # build dataset & dataloader
            dataset = build_dataset(cfg.data.test)
            data_loader = build_dataloader(
                    dataset,
                    samples_per_gpu=samples_per_gpu,
                    workers_per_gpu=cfg.data.workers_per_gpu,
                    dist=False,
                    shuffle=False)

            output = single_gpu_test(model, data_loader, show_score_thr=0) # output 계산

            # submission 양식에 맞게 output 후처리
            prediction_strings = []
            file_names = []
            coco = COCO(cfg.data.test.ann_file)

            class_num = 10
            if json_file == 'test.json':
                for i, out in enumerate(output):
                    prediction_string = ''
                    image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
                    for j in range(class_num):
                        for o in out[j]:
                            if f_r == 1.0:
                                o_copy = o.copy()
                                o[0] = 1024 - o_copy[2]
                                o[2] = 1024 - o_copy[0]
                            prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                                o[2]) + ' ' + str(o[3]) + ' '
                        
                    prediction_strings.append(prediction_string)
                    file_names.append(image_info['file_name'])

            else:
                for i, out in enumerate(output):
                    prediction_string = ''
                    for j in range(class_num):
                        for o in out[j]:
                            if f_r == 1.0:
                                o_copy = o.copy()
                                o[0] = 1024 - o_copy[2]
                                o[2] = 1024 - o_copy[0]
                            prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(
                                o[2]) + ' ' + str(o[3]) + ' '
                        
                    prediction_strings.append(prediction_string)
                    file_names.append(dataset[i]['img_metas'][0].data['ori_filename'])

            submission = pd.DataFrame()
            submission['PredictionString'] = prediction_strings
            submission['image_id'] = file_names
            flip_name = 'ori' if f_r==0.0 else 'flip'

            if not os.path.isdir(os.path.join(cfg.work_dir, f'tta')):
                os.makedirs(os.path.join(cfg.work_dir, f'tta'))

            if json_file == 'test.json':
                submission.to_csv(os.path.join(cfg.work_dir, f'tta/submission_{scale[0]}_{flip_name}.csv'), index=None)
            else:
                submission.to_csv(os.path.join(cfg.work_dir, f'tta/valid_{scale[0]}_{flip_name}.csv'), index=None)
            # submission.head()

if __name__ == "__main__":
    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    scales = [(768, 768), (672, 672), (576, 576), (480, 480)]

    # json_file = 'split_valid.json'
    json_file = 'test.json'
    config = './configs/custom_config/swin_transformer_pseudo.py'
    weight = 'epoch_46'
    work_dir = './work_dirs/pseudo_label/swin_transformer/'
    samples_per_gpu = 32
    

    main(config, scales, img_norm_cfg, weight, work_dir, samples_per_gpu, json_file=json_file)
