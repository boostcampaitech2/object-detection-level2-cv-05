import pandas as pd
from ensemble_boxes import *
import numpy as np
from pycocotools.coco import COCO


def main(submission_files, mode='wbf', weights=None, json_file='test.json', name=None):
    if mode == 'wbf':
        weights = weights
    else:
        assert mode == 'nms' or mode == 'softnms'
        weights = [0]*len(submission_files)

    submission_df = [pd.read_csv(file) for file in submission_files]
    image_ids = submission_df[0]['image_id'].tolist()

    annotation = '../dataset/' + json_file 
    # annotation = '../dataset/split_valid.json'
    coco = COCO(annotation)

    prediction_strings = []
    file_names = []
    iou_thr = 0.5

    for i, image_id in enumerate(image_ids):
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []
        if not json_file == 'test.json':
            i = int(image_id[6:10])
        image_info = coco.loadImgs(i)[0]
        
        image_weights = []

        for df, weight in zip(submission_df, weights):
            predict_string = df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
            predict_list = str(predict_string).split()
            
            if len(predict_list)==0 or len(predict_list)==1:
                continue
                
            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []
            
            for box in predict_list[:, 2:6].tolist():
                box[0] = float(box[0]) / image_info['width']
                box[1] = float(box[1]) / image_info['height']
                box[2] = float(box[2]) / image_info['width']
                box[3] = float(box[3]) / image_info['height']
                box_list.append(box)
                
            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))
            image_weights.append(weight)
    
        if len(boxes_list):
            if mode == 'wbf':
                boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=image_weights, iou_thr=iou_thr)
            elif mode == 'nms':
                boxes, scores, labels = nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr)
            elif mode == 'softnms':
                boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr)

            labels = labels.astype(np.int32)

            for box, score, label in zip(boxes, scores, labels):
                prediction_string += str(label) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '
    
        prediction_strings.append(prediction_string)
        file_names.append(image_id)

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    if name:
        submission.to_csv(name)
    elif json_file == 'test.json':
        submission.to_csv('submission_ensemble.csv')
    else:
        name = json_file.split('.')[0]
        submission.to_csv(f'{name}_ensemble.csv')
    

if __name__ == "__main__":
    json_file = 'split_valid.json'
    # json_file='test.json'
    # mode = 'nms'
    mode = 'wbf'
    submission_files = [
                        './work_dirs/pseudo_label/swin_transformer/tta/valid_480_ori.csv', 
                        './work_dirs/pseudo_label/swin_transformer/tta/valid_480_flip.csv',
                        ]



    weights=[1,1]
    main(submission_files, mode=mode, weights=weights, json_file=json_file, name=None)