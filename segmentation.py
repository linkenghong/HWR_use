import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from skimage import measure, color
from prediction import image_prediction
from scipy.ndimage import gaussian_filter1d
import time

# 利用图片分块预测的先验知识确定最佳图片分割点
def find_seg_point(seg_num, char_num, confM, charConfM, seg_type, index, count, alt_seg, max_conf, max_seg, label_weight = 5):
    if count == 1:
        temp = 0
        alt_seg.append(seg_num-1)
        for i in range(char_num):
            # 对连通域求最佳分割点，分割列表alt_seg中为一个分割区域包含的最后一个连通区域
            if seg_type == 2:
                temp += confM[alt_seg[i]+1][alt_seg[i+1]] + label_weight * charConfM[alt_seg[i]+1][alt_seg[i+1]][i]
            # 对极值点列表求最佳分割点，分割列表中包含分割点
            if seg_type == 3:
                temp += confM[alt_seg[i]][alt_seg[i+1]] + label_weight * charConfM[alt_seg[i]][alt_seg[i+1]][i]
        if temp > max_conf:
            max_conf = temp
            max_seg = alt_seg.copy()
        alt_seg.pop()
    else:
        for end in range(index, seg_num - count + 1):
            alt_seg.append(end)
            alt_seg, max_conf, max_seg = find_seg_point(seg_num, char_num, confM, charConfM, seg_type, end+1, count-1, alt_seg, max_conf, max_seg, label_weight = label_weight)
    if len(alt_seg):
        alt_seg.pop()
    return alt_seg, max_conf, max_seg


# 投影法：按行扫描，获取每个字符起始行终止行
def extrack_point(line_sum, min_sum, min_gap):
    # 按行或列扫描，得到大于阈值min_sum的索引
    nozero_index = np.where(line_sum>min_sum)[0]
    extrack_point = []
    pre_p = None
    for cur_p in nozero_index:
        if pre_p == None:
            start_p = cur_p
            pre_p = cur_p
            continue
        # 索引间大于间隔阈值min_gap的才算一个新字符开始
        elif cur_p - pre_p > min_gap:
            end_p = pre_p
            extrack_point.append((start_p, end_p))
            start_p = cur_p
        pre_p = cur_p
    extrack_point.append((start_p, pre_p))
    return extrack_point

def find_best_borders_projection(img, name, threshold = 200, direction = 0, min_sum = 1, min_gap = 1):
    limg = img.convert('L').point(lambda p: 0 if p > threshold else 1, '1')
    img_data = np.array(limg, dtype=np.int32)
    verdire_sum = img_data.sum(axis=1-direction)
    length = len(np.where(verdire_sum>0)[0])
    char_num = len(name)

    extrack_points = extrack_point(verdire_sum, min_sum, min_gap)
    if len(extrack_points) != char_num:
        return []

    char_min = length / char_num / 2
    char_max = min(length / char_num * 2, length / 3 * 2)
    for point in extrack_points:
        if point[1] - point[0] < char_min or point[1] - point[0] > char_max:
            return []

    direction_sum = img_data.sum(axis=direction)
    up, down = np.where(direction_sum>0)[0][0], np.where(direction_sum>0)[0][-1]
    borders = []
    for point in extrack_points:
        if direction == 0:
            borders.append([up, point[0], down, point[1]])
        elif direction == 1:
            borders.append([point[0], up, point[1], down])
    
    return borders


# 连通域法：
def find_connect(img, direction = 0, threshold = 200, overlap = 0.5):
    # 将图片转化为反相二值图，即将原来较白的（大于阈值，越大越白）转为0，较黑的（低于阈值的）转为1
    limg = img.convert('L').point(lambda p: 0 if p > threshold else 1, '1')
    limg = np.array(limg, dtype=np.int32)
    labels, num = measure.label(limg, connectivity=1, return_num = True)
    # dst = color.label2rgb(labels)
    # plt.imshow(dst)
    label_areas = []
    for i in range(1, num+1):
        label_areas.append(
            [min(np.where(labels==i)[direction]),
            max(np.where(labels==i)[direction])]
            )
    label_areas.sort(key=lambda x:x[0])
    merge_areas = []
    for area in label_areas:
        if (not merge_areas or merge_areas[-1][1] < area[0] 
            or merge_areas[-1][1] - area[0] < min(area[1]-area[0], merge_areas[-1][1]-merge_areas[-1][0])*overlap):
            merge_areas.append(area)
        else:
            merge_areas[-1][1] = max(merge_areas[-1][1], area[1])
    
    return merge_areas

def find_best_borders_connect(img, name, threshold = 200, direction = 0, overlap = 0.85, net_type = 'S'):
    limg = img.convert('L').point(lambda p: 0 if p > threshold else 1, '1')
    img_data = np.array(limg, dtype=np.int32)
    direction_sum = img_data.sum(axis=direction)
    up, down = np.where(direction_sum>0)[0][0], np.where(direction_sum>0)[0][-1]
    verdire_sum = img_data.sum(axis=1-direction)
    length = np.where(verdire_sum>0)[0][-1] - np.where(verdire_sum>0)[0][0]
    merge_area = find_connect(img, direction = direction, overlap = overlap)
    area_num = len(merge_area)
    char_num = len(name)
    name_list = list(name)
    char_min = length / char_num / 3
    char_max = min(length / char_num * 2, length / 3 * 2)

    # 连笔严重，无法切割出合适的连通区域
    if char_num > area_num:
        return []
    elif char_num == area_num:
        for area in merge_area:
            if (area[1] - area[0] <= char_min or area[1] - area[0] >= char_max ):
                return []
        max_seg = [-1] + [i for i in range(char_num)]
    else:
        for area in merge_area:
            if area[1] - area[0] >= char_max:
                return []
        confM = [[0 for i in range(area_num)] for j in range(area_num)]
        charM = [[' ' for i in range(area_num)] for j in range(area_num)]
        charConfM = [[[0] * char_num for i in range(area_num)] for j in range(area_num)]
        for i in range(area_num):
            for j in range(i, area_num):
                if (merge_area[j][1] - merge_area[i][0] >= char_min and
                    merge_area[j][1] - merge_area[i][0] <= char_max ):
                    if direction == 0:
                        border = [up, merge_area[i][0], down, merge_area[j][1]]
                    elif direction == 1:
                        border = [merge_area[i][0], up, merge_area[j][1], down]
                    pred_img = img.crop(border)
                    # display(pred_img)
                    charM[i][j], confM[i][j], _, charConfM[i][j] = image_prediction(pred_img, topk = 1, labels = name_list, net_type = net_type)

        _, max_conf, max_seg = find_seg_point(area_num, char_num, confM, charConfM, 2, 0, char_num, [-1], 0, [-1])

        if len(max_seg) != char_num + 1:
            return []
    borders = []
    for i in range(char_num):
        if direction == 0:
            borders.append([up, merge_area[max_seg[i]+1][0], down, merge_area[max_seg[i+1]][1]])
        elif direction == 1:
            borders.append([merge_area[max_seg[i]+1][0], up, merge_area[max_seg[i+1]][1], down])
        
    return borders

# 求数列的极值点
def find_local_min(arr):
    loc_min = []
    loc_min.append(np.where(arr > 0)[0][0])
    pre = arr[0]
    alt_min_pos = None
    alt_min = None
    for i, a in enumerate(arr):
        if a < pre:
            alt_min_pos = i
            alt_min = a
        elif a > pre and alt_min_pos:
            if pre == alt_min:
                alt_min_pos = (i - 1 + alt_min_pos) // 2
            loc_min.append(alt_min_pos)
            alt_min_pos = None
            alt_min = None
        pre = a
    loc_min.append(np.where(arr > 0)[0][-1])
    return loc_min

# 投影高斯平滑极值法
def find_best_borders_gas(img, name, threshold = 200, direction = 0, gas_std = 8, net_type = 'S'):
    name_list = list(name)
    char_num = len(name)
    limg = img.convert('L').point(lambda p: 0 if p > threshold else 1, '1')
    img_data = np.array(limg, dtype=np.int32)
    direction_sum = img_data.sum(axis=direction)
    up, down = np.where(direction_sum>0)[0][0], np.where(direction_sum>0)[0][-1]
    verdire_sum = img_data.sum(axis=1-direction)
    gas8 = gaussian_filter1d(verdire_sum, gas_std)
    local_min = find_local_min(gas8)
    length = local_min[-1] - local_min[0]
    min_num = len(local_min)
    char_min = length / char_num / 2
    char_max = min(length / char_num * 2, length / 3 * 2)
    count = 0
    if min_num-1 < char_num:
        return []
    elif min_num-1 == char_num:
        for i in range(1, min_num):
            if local_min[i] - local_min[i-1] <= char_min or local_min[1] - local_min[0] >= char_max:
                return []
        max_seg = [i for i in range(min_num)]
    else:
        confM3 = [[0 for i in range(min_num)] for j in range(min_num)]
        charM3 = [[' ' for i in range(min_num)] for j in range(min_num)]
        charConfM3 = [[[0] * char_num for i in range(min_num)] for j in range(min_num)]
        for i in range(min_num):
            for j in range(i+1, min_num):
                if(local_min[j] - local_min[i] >= char_min and local_min[j] - local_min[i] <= char_max):
                    if direction == 0:
                        border = [up, local_min[i], down, local_min[j]]
                    elif direction == 1:
                        border = [local_min[i], up, local_min[j], down]
                    pred_img = img.crop(border)
                    # display(pred_img)
                    count += 1
                    charM3[i][j], confM3[i][j], toplist, charConfM3[i][j] = image_prediction(pred_img, topk = 1, labels=name_list, net_type = net_type)
        _, max_conf, max_seg = find_seg_point(min_num, char_num, confM3, charConfM3, 3, 0, char_num, [0], 0, [0])
        if len(max_seg) != char_num + 1:
            return []

    # print(name, ", minpoint count: ", min_num, ", predict count: ", count)
    borders = []
    for i in range(char_num):
        if direction == 0:
            borders.append([up, local_min[max_seg[i]], down, local_min[max_seg[i+1]]])
        elif direction == 1:
            borders.append([local_min[max_seg[i]], up, local_min[max_seg[i+1]], down])
    
    return borders


def show_borders(img, borders, width = 4):
    img_c = img.copy()
    draw =ImageDraw.Draw(img_c)
    for border in borders:
        draw.rectangle(border, outline='red', width=width)
    plt.imshow(img_c)
    plt.show()

def show_gas(img, threshold = 200, direction = 0, gas_std = 8):
    limg = img.convert('L').point(lambda p: 0 if p > threshold else 1, '1')
    img_data = np.array(limg, dtype=np.int32)
    verdire_sum = img_data.sum(axis=1-direction)
    gas8 = gaussian_filter1d(verdire_sum, gas_std)
    plt.bar(range(len(verdire_sum)), verdire_sum, width=1)
    plt.show()
    plt.plot(gas8, '--', label='filtered, sigma=8')
    plt.legend()
    plt.show()
    if direction == 0:
        plt.imshow(img.rotate(90,expand = True))
    else:
        plt.imshow(img)
    plt.show()
    