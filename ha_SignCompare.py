from PIL import Image
from segmentation import find_best_borders_connect, find_best_borders_projection, find_best_borders_gas, show_borders
from prediction import image_prediction
import sys
from glob import glob
from alfred.utils.log import logger as logging


def ha_SignCompare(image, text):
    img = Image.open(image).convert('RGB')
    direction = 0 if img.size[0] < img.size[1] else 1
    name_num = len(text)
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