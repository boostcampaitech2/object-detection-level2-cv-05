import pandas as pd
import numpy as np

from pycocotools.coco import COCO


CLASSES = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

def process_result_csv(csv_file):
    result_df = pd.read_csv(csv_file)

    file_names = result_df['image_id'].values.tolist()
    bboxes = result_df['PredictionString'].fillna(-100).values.tolist()

    result = []
    for file_name, bbox in zip(file_names, bboxes):
        if bbox== -100:
            continue
        bbox = bbox.split()
        for slc in range(len(bbox)//6):
            label, confid, x_min, y_min, x_max, y_max = bbox[slc*6:(slc+1)*6]
            label = int(label)
            confid = float(confid)
            x_min, y_min, x_max, y_max = map(int, map(float, [x_min, y_min, x_max, y_max]))
            result.append([file_name, CLASSES[label], confid , x_min, x_max, y_min, y_max])

    return np.array(result)

def process_gt_json(json_file):
    coco = COCO(json_file)
    ID_list = coco.getImgIds()

    result = []
    for image_id in ID_list:
        image_info = coco.loadImgs(image_id)[0]
        file_name = image_info['file_name']
        bbox_ids = coco.getAnnIds(image_info['id'])
        for bbox_id in bbox_ids:
            bbox_anno = coco.loadAnns(bbox_id)[0]
            xm,ym,w,h = list(map(int, bbox_anno['bbox']))
            norm_bbox = [xm,xm+w, ym, ym+h]
            bbox_info_list = [file_name] + [CLASSES[bbox_anno['category_id']]]+norm_bbox
            result.append(bbox_info_list)
    
    return np.array(result)

def sort_by_confidence(pred_array, category=None):

    if category is not None:
        if isinstance(category, str):
            assert category in CLASSES, "Wrong category name"
        if isinstance(category, int):
            assert category in range(len(CLASSES)), "Wrong category index"
            category = CLASSES[category]
    
    if category:
        query_inds = np.where(pred_array[:,1]==category)[0]
        pred_array = pred_array[query_inds]
    
    return sorted(pred_array, key=lambda x : float(x[2]), reverse=True)

