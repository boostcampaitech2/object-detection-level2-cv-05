# object-detection-level2-cv-05

## Project 개요
- 목표 : 쓰레기를 분리수거 종류별로 분류하기
  - General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing 10 종류의 쓰레기
- Data
  - Training Data : 쓰레기별로 bounding box와 category가 labeling 되어있는 4883장의 데이터
  - Test Data : 무작위로 선정된 4871장의 데이터
- 분포 
  - bbox 크기 불균형 문제 : 크기에 따라 bbox를 small/medium/large 3가지 카테고리로 나누었을 때, small과 medium 크기의 bbox는 large bbox에 비해 데이터 수가 현저히 작은 것을 확인할 수 있었다. (small : ~322, medium : ~962,  large : 962~ )
  - Class 불균형 문제 : battery와 clothing 클래스가 적고, paper와 plastic bag 클래스가 많은 데이터 불균형 문제가 있었다. 
  - General Trash 문제 : General Trash에 해당하는 쓰레기의 종류가 다양하고 general trash에 해당하는 object들 중 small/medium 크기의 object 비율이 높아 general trash를 잘 분류하지 못하는 문제점이 있었다. 

## Table of Contents
1. [Train](#Train)
2. [Code Structure](#code-structure)
3. [Detail](#detail)
4. [Contributor](#contributor)


### Result
- Private mAP50 score: 0.661



## Getting Started
```bash
pip install -r requirements.txt
```

### Train
```bash
cd mmdetection
python tools/train.py configs/...
```
```bash
cd yolov5
python train.py --cfg ...
```
- 사용한 config 목록

| model                                | augment                              | LB score(mAP 50) | config file                   |
|--------------------------------------|:------------------------------------:|:----------------:|:-----------------------------:|
| cascade - swin large                 | multi-scale                          | 0.620            |[config](https://github.com/boostcampaitech2/object-detection-level2-cv-05/blob/main/configs/cascade_swin/cascade_swin.py)|
| cascade - swin tiny (pseudo labeling)| multi-scale                          | 0.595            |[config](https://github.com/boostcampaitech2/object-detection-level2-cv-05/blob/main/configs/pseudo_swin_transformer/pseudo_swin_transformer.py)|
| cascade - swin large                 | multi-scale, copypasting             | 0.561            |[config](https://github.com/boostcampaitech2/object-detection-level2-cv-05/tree/main/configs/cascade_swin_fpn_copypasting/swin_transformer.py)                               |
| detectoRS - resnet101                | multi-scale                          | 0.525            |[config](https://github.com/boostcampaitech2/object-detection-level2-cv-05/blob/main/configs/detectors/detectors_htc_r101_rfp.py)|
| vfnet - swin large                   | multi-scale ,high-resolution, mosaic | 0.540            |[config](https://github.com/boostcampaitech2/object-detection-level2-cv-05/blob/main/configs/vfnet/vfnet_swin_large_pafpn.py)|
| yolov5x6 - csp darknet               | multi-scale,autoAugment              | 0,524            |[config](https://github.com/boostcampaitech2/object-detection-level2-cv-05/blob/main/yolov5/models/trash_yolov5x6.yaml)|



### Inference
```bash
cd mmdetection
python tta_inference.py
python ensemble.py
```
```bash
cd yolov5
python detect.py --augment
```



## Code Structure
```
├── mmdetection                    # code from mmdetection
│   ├── tools/train.py             # to train 
│   ├── tta_inference.py           # test time augmentation inference 
│   └── ensemble.py                # ensemble ( weighted nms )
└── yolov5                         # code from yolov5
    ├── train.py                   # to train 
    └── detect.py                  # inference
```


## Detail
---
### Model
- <a href = 'https://github.com/open-mmlab/mmdetection'>mmdetection</a>과 <a href = 'https://github.com/ultralytics/yolov5'>yolov5</a>를 기반으로 실험을 진행하였다.
- 효율적으로 ensemble을 하기 위해서 다양한 model을 사용하려고 노력하였다.



### Dataset
- small/median object에 robust한 모델을 만들기 위해 작은 object bbox들을 copy해 image내 random한 위치에 bbox를 paste하는 copypasting augmentation을 사용했다.
- Mosaic, Cutout, Multi-scale 을 활용해 일반화된 학습이 되도록 하였다. 
- Pseudo Labeling: 앙상블한 test dataset 결과를 이용해 pseudo label을 생성하여 사용하였다. confidence에 따라 loss에 weight를 주어 학습에 사용하였다. (swin cascade tiny backbone 기준으로 LB score 0.011 상승)




## Contributor
- 강수빈([github](https://github.com/suuuuuuuubin)) : 2 Stage model 학습
- 김인재([github](https://github.com/K-nowing)) : 데이터 분석 및, pseudo labeling 등 다양한 실험 진행
- 원상혁([github](https://github.com/wonsgong)) : 1 Stage model 학습
- 이경민([github](https://github.com/lkm2835)) : 2 Stage model, small object detection
- 최민서([github](https://github.com/minseo0214)) : 데이터 분석 및, 2 Stage model 학습


