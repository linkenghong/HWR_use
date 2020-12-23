from PIL import Image
from segmentation import find_best_borders_connect, find_best_borders_projection, find_best_borders_gas, show_borders
from prediction import image_prediction
import numpy as np
import sys
from glob import glob
from alfred.utils.log import logger as logging


def ha_SignCompare(image, text):
    img = Image.open(image).convert('RGB')

    threshold = 200
    limg = img.convert('L').point(lambda p: 0 if p > threshold else 1, '1')
    img_data = np.array(limg, dtype=np.int32)
    dire_sum_0 = img_data.sum(axis=0)
    dire_sum_1 = img_data.sum(axis=1)
    no0_dir0 = np.where(dire_sum_0>0)[0]
    no0_dir1 = np.where(dire_sum_1>0)[0]
    direction = 1
    if no0_dir0[-1] - no0_dir0[0] < no0_dir1[-1] - no0_dir1[0]:
        direction = 0

    code, message, char_count, char_list = 0, '', 0, []
    if direction == 0:
        borders = find_best_borders_projection(img, text, direction = direction, min_sum=1, min_gap=1)
        if borders:
            for border in borders:
                pred_char, conf, toplist = image_prediction(img.crop(border), topk = 30)
                char_list.append(toplist)
        else:
            borders = find_best_borders_connect(img, text, direction = direction)
            if borders:
                for border in borders:
                    pred_char, conf, toplist = image_prediction(img.crop(border), topk = 30)
                    char_list.append(toplist)
            else:
                borders = find_best_borders_gas(img, text, direction = direction)
                if borders:
                    for border in borders:
                        pred_char, conf, toplist = image_prediction(img.crop(border), topk = 30)
                        char_list.append(toplist)

    else:
        borders = find_best_borders_gas(img, text, direction = direction)
        if borders:
            for border in borders:
                pred_char, conf, toplist = image_prediction(img.crop(border), topk = 30)
                char_list.append(toplist)
    # show_borders(img, borders)
    if char_list:
        return 0, 'Successful', len(char_list), char_list
    else:
        return -1, 'Failed', 0, []

if __name__ == "__main__":
    pass