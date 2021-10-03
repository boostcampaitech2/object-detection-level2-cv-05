from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)

from pathlib import Path
from importlib import import_module
import os
import argparse

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

    return path_


def main(args):
    # config file 들고오기
    root = 'configs'
    cfg = Config.fromfile(f'{root}/{args.config}')

    # dataset config 수정
    cfg.work_dir = increment_path(args.work_dir)

    # config 저장
    _base_ = getattr(import_module(f"{root}.{args.config.split('.')[0]}"), '_base_')
    for base in _base_:
        os.system(f'cp {root}/{base} {cfg.work_dir}')
    
    # wandb 연동
    SAVENAME = cfg.model.type +'_'+ cfg.model.backbone.type +'_'+ cfg.model.neck.type
    cfg.log_config.hooks.append(
        dict(type='WandbLoggerHook',
                init_kwargs = dict(project = "Trash_Detect",
                                    entity = 'friends',
                                    name = SAVENAME,
                                    config = dict(batch_size = cfg.data.samples_per_gpu,
                                                lr           = cfg.optimizer.lr,
                                                epochs       = cfg.runner.max_epochs,
                                                model        = cfg.model.type,
                                                save_name    = SAVENAME
                                                 )
                                  )
            )
    )


    # build_dataset
    datasets = [build_dataset(cfg.data.train)]

    # dataset 확인
    datasets[0]

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Config')
    args.add_argument('-c', '--config', default=None, type=str, help='config file name (default: None)')
    args.add_argument('-w', '--work_dir', default='work_dirs/exp', type=str, help='work_dir (default: work_dirs/exp)')

    args = args.parse_args()
    msg_no_cfg = "Configuration file need to be specified. Add '-c faster_rcnn_r50_fpn_1x_coco.py', for example"
    assert args.config is not None, msg_no_cfg

    main(args)