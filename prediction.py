import pickle
import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from PIL import Image, ImageDraw, ImageFont
from model import HWDB_GoogLeNet


def image_prediction(img, net_num = 825, topk = 30, labels = None):
    with open('char_dict', 'rb') as f:
        char_dict = pickle.load(f)
    index_char_dict = {index: key for key, index in char_dict.items()}
    num_classes = len(char_dict)

    transform = transforms.Compose([
        transforms.Resize((120, 120)),
        transforms.ToTensor(),
    ])
    input = transform(img)
    input = input.unsqueeze(0)

    net = HWDB_GoogLeNet(num_classes)
    pth_dir = r'./checkpoints'
    net_path = os.path.join(pth_dir, 'HWDB_GoogLeNet_iter_' + str(net_num) +'.pth')
    if torch.cuda.is_available():
        net = net.cuda()
        input = input.cuda()
    net.load_state_dict(torch.load(net_path))
    net.eval()

    output = net(input)
    output_sm = nn.functional.softmax(output, dim=1)

    topk_list = torch.topk(output_sm, topk)[1].tolist()[0]
    topk_char = [index_char_dict[ind] for ind in topk_list]
    conf, pred = torch.max(output_sm.data, 1)
    pred_char = index_char_dict[pred.item()]
    conf = conf.item()

    if labels == None:
        return pred_char, conf, topk_char
    else:
        label_confs = []
        for label in labels:
            try:
                label_conf = output_sm.tolist()[0][char_dict[label]]
                label_confs.append(label_conf)
            except:
                label_confs.append(0)
        return pred_char, conf, topk_char, label_confs







