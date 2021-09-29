from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)

from pathlib import Path
import json
import os

def increment_path(path):   
    n = 0
    while True:
        path_ = Path(f"{path}{n}")
        if not path_.exists():
            break
        elif path_.exists():
            n += 1

    path_ = str(path_)
    path = ''
    for p in path_.split('/'):
        path += f'{p}/'
        if not Path(path).exists():
            os.mkdir(path)

    print(path_)
    return path_

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile('/opt/ml/detection/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
#cfg = Config.fromfile('/opt/ml/detection/mmdetection/configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py')

root='/opt/ml/detection/dataset/'

# dataset config 수정
resize = (1024, 1024)
cfg.data.train.classes = classes
cfg.data.train.img_prefix = root
cfg.data.train.ann_file = root + 'train.json' # train json 정보
cfg.data.train.pipeline[2]['img_scale'] = resize # Resize

'''
cfg.data.val.classes = classes
cfg.data.val.img_prefix = root
cfg.data.val.ann_file = root + '' #valid json 정보
cfg.data.val.pipeline[1]['img_scale'] = resize
'''

cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'test.json' # test json 정보
cfg.data.test.pipeline[1]['img_scale'] = resize # Resize

cfg.data.samples_per_gpu = 4

cfg.seed = 2021
cfg.gpu_ids = [0]
cfg.work_dir = increment_path('./work_dirs/faster_rcnn/exp')
cfg.model.roi_head.bbox_head.num_classes = 10
#cfg.model.rpn_head.anchor_generator.ratios = [0.55, 1.0, 1.8]
#cfg.model.rpn_head.loss_cls = {
#    "type": "FocalLoss",
#    "use_sigmoid": True,
#    "loss_weight":1.0
#}

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)

cfg.runner.max_epochs = 10

# config 저장
saved_cfg = dict()
for key, value in cfg.items():
    saved_cfg[key] = value

with open(f"{cfg.work_dir}/config.json", "w") as json_file:
    saved_cfg = json.dumps(saved_cfg, indent=4)
    json_file.write(saved_cfg)

# build_dataset
datasets = [build_dataset(cfg.data.train)]

# dataset 확인
datasets[0]

# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()


# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=False)