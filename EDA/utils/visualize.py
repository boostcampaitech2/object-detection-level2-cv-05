from PIL import Image
import os
import numpy as np

import bbox_visualizer as bbv


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
    image_array = bbv.draw_rectangle(image_array,(x_1, y_1, x_2, y_2))
    image_array = bbv.add_label(image_array, label_name, (x_1, y_1, x_2, y_2))
    
    image = Image.fromarray(image_array)
    if save_path:
        image.save(save_path)
    return image