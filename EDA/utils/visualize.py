from PIL import Image
import os
import numpy as np

import bbox_visualizer as bbv

COLORS =   {"General trash" : (152, 223, 138),
            "Paper" : (174, 199, 232),
            "Paper pack" : (31, 119, 180),
            "Metal" : (255, 152, 150),
            "Glass" : (247, 182, 210),
            "Plastic" : (140, 86, 75),
            "Styrofoam" : (82, 84, 163),
            "Plastic bag" : (255, 187, 120),
            "Battery" : (197, 176, 213),
            "Clothing" : (214, 39, 40)}

def draw_one_bbox(info, image_root='/opt/ml/detection/dataset', save_path = None):
    """
    info: [file_path, label_name, confidence, x_1, x_2, y_1, y_2]
        or [file_path, label_name x_1, x_2, y_1, y_2]
    """
    if len(info) == 7:
        file_path, label_name, confidence, x_1, x_2, y_1, y_2 = info
    elif len(info) == 6:
        file_path, label_name, x_1, x_2, y_1, y_2 = info
    else:
        raise
    if isinstance(x_1, str):
        x_1 = float(x_1)
        x_2 = float(x_2)
        y_1 = float(y_1)
        y_2 = float(y_2)
    if x_1%1 !=0:
        x_1 = x_1*1024
        x_2 = x_2*1024
        y_1 = y_1*1024
        y_2 = y_2*1024    
    x_1 = int(x_1)
    x_2 = int(x_2)
    y_1 = int(y_1)
    y_2 = int(y_2)
    image = Image.open(os.path.join(image_root, file_path))
    image_array = np.array(image)
    image_array = bbv.draw_rectangle(image_array,(x_1, y_1, x_2, y_2), bbox_color=COLORS[label_name])
    image_array = bbv.add_label(image_array, label_name, (x_1, y_1, x_2, y_2), text_bg_color=COLORS[label_name])
    
    image = Image.fromarray(image_array)
    if save_path:
        image.save(save_path)
    return image

def draw_multi_bbox(infos, image_root='/opt/ml/detection/dataset', save_path = None, confidence_thresh=None):
    """
    infos: [info_0, info_1, info_2 ...]
    confidence_thresh: you can use this parameter, when the infos are predictions.
    """
    assert all(infos[:,0] == infos[0,0]), "All file pathes should be the same."

    file_path = infos[0][0]
    image = Image.open(os.path.join(image_root, file_path))
    image_array = np.array(image)

    for info in infos:
        if len(info) == 7:
            file_path, label_name, confidence, x_1, x_2, y_1, y_2 = info
        elif len(info) == 6:
            file_path, label_name, x_1, x_2, y_1, y_2 = info
        else:
            raise
        if confidence_thresh:
            if float(confidence) < confidence_thresh:
                continue
        if isinstance(x_1, str):
            x_1 = float(x_1)
            x_2 = float(x_2)
            y_1 = float(y_1)
            y_2 = float(y_2)
        if x_1%1 !=0:
            x_1 = x_1*1024
            x_2 = x_2*1024
            y_1 = y_1*1024
            y_2 = y_2*1024    
        x_1 = int(x_1)
        x_2 = int(x_2)
        y_1 = int(y_1)
        y_2 = int(y_2)
        image_array = bbv.draw_rectangle(image_array,(x_1, y_1, x_2, y_2), bbox_color=COLORS[label_name])
        image_array = bbv.add_label(image_array, label_name, (x_1, y_1, x_2, y_2), text_bg_color=COLORS[label_name])
    
    image = Image.fromarray(image_array)
    if save_path:
        image.save(save_path)
    return image