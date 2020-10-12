# -*- coding: utf-8 -*-
# @File : config.py
# @Author: Runist
# @Time : 2020-09-11 13:54
# @Software: PyCharm
# @Brief: 配置文件

rpn_lr_max = 1e-4
rpn_lr_min = 1e-9
cls_lr_max = 1e-5
cls_lr_min = 1e-9
lr = 1e-4

epoch = 100

anchor_box_scales = [128, 256, 512]
anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]

batch_size = 1
rpn_stride = 16
input_shape = (600, 600)
share_layer_shape = (round(input_shape[0] / rpn_stride), round(input_shape[1] / rpn_stride))


num_rois = 128
num_regions = 256
valid_rate = 0
shuffle_size = 1024

rpn_min_overlap = 0.3
rpn_max_overlap = 0.7
classifier_min_overlap = 0.1
classifier_max_overlap = 0.5
classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

data_pretreatment = 'random'

annotation_path = "./config/train.txt"
weight_path = './logs/model/faster_rcnn.h5'
summary_path = './logs/summary/'
label = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

num_classes = len(label) + 1
