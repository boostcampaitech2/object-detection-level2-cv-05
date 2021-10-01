import numpy as np

from map_boxes import mean_average_precision_for_boxes

from .process import process_result_csv, process_gt_json

def IoU(box1, box2):
    # box = (x1, x2, y1, y2)
    box1_area = (box1[1] - box1[0] + 1) * (box1[3] - box1[2] + 1)
    box2_area = (box2[1] - box2[0] + 1) * (box2[3] - box2[2] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    x2 = min(box1[1], box2[1])
    y1 = max(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def get_mAP(pred, gt, iou_threshold=0.5, verbose=False):
    
    if isinstance(pred, str):
        pred = process_result_csv(pred)
    if isinstance(gt, str):
        gt = process_gt_json(gt)

    mAP, average_precisions = mean_average_precision_for_boxes(gt, pred, iou_threshold=iou_threshold, verbose=verbose)
    return mAP, average_precisions