import json
import cv2
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

JSON_PATH  = '../detection/dataset/train.json'
CSV_PATH   = '../detection/faster_rcnn/faster_rcnn_torchvision_submission.csv'
IMAGE_PATH = '../detection/dataset/'
SAVE_PATH  = './bbox_data/'
MIN_SCORE = 0.85


with open(JSON_PATH, 'r') as json_file:
    train_json = json.load(json_file)

train_images = train_json['images']
train_categories = train_json['categories']
train_annotations = train_json['annotations']

try:
    os.mkdir(f'{SAVE_PATH}')
except FileExistsError as e:
    print(e)

try:
    os.mkdir(f'{SAVE_PATH}/test')
except FileExistsError as e:
    print(e)

colors =   [[152, 223, 138],
            [174, 199, 232],
            [31, 119, 180],
            [255, 152, 150],
            [247, 182, 210],
            [140, 86, 75],
            [82, 84, 163],
            [255, 187, 120],
            [197, 176, 213],
            [214, 39, 40],
            [255, 127, 14],
            [23, 190, 207],
            [44, 160, 44],
            [112, 128, 144]]

df = pd.read_csv(CSV_PATH)

for i in range(len(df)):
    prediction_str = df['PredictionString'][i]
    if type(prediction_str) == float:
        continue
    prediction_str = prediction_str.split(' ')
    idx = 0

    file_name = df['image_id'][i]
    img = cv2.imread(f"{IMAGE_PATH}{file_name}")
    while idx+5 < len(prediction_str):
        label = int(prediction_str[idx])
        score = float(prediction_str[idx+1])
        if score < MIN_SCORE:
            break
        xmin  = int(float(prediction_str[idx+2]))
        ymin  = int(float(prediction_str[idx+3]))
        xmax  = int(float(prediction_str[idx+4]))
        ymax  = int(float(prediction_str[idx+5]))

        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)

        img = cv2.rectangle(img, pt1, pt2, colors[label], 3)
        font = cv2.FONT_HERSHEY_PLAIN
        name_score = f"{train_categories[label]['name']} {score*100:.1f}"
        cv2.putText(img, name_score, pt1, font, 2, (255,255,255), 2, cv2.LINE_4)
        idx += 6

    cv2.imwrite(f"{SAVE_PATH}{file_name}", img)