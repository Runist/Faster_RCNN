# -*- coding: utf-8 -*-
# @File : anchorGenerate.py
# @Author: Runist
# @Time : 2020/9/9 20:53
# @Software: PyCharm
# @Brief: 生成先验框

import numpy as np
import config.config as cfg


def generate_anchors(sizes=None, ratios=None):
    """
    生成先验框的9种尺寸
    :param sizes: 先验框的尺寸
    :param ratios: 先验框的长宽比
    :return: 9种尺寸的先验框
    """
    if ratios is None:
        ratios = cfg.anchor_box_ratios
    if sizes is None:
        sizes = cfg.anchor_box_scales

    num_anchors = len(sizes) * len(ratios)

    # 每个点9个先验框，4个坐标
    anchors = np.zeros((num_anchors, 4), dtype=np.float32)

    # 将size的内容在，0维度上重复一次，1维度上重复3次
    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T
    # 那么此时anchor中的内容就是
    # [[  0.   0. 128. 128.]
    #  [  0.   0. 256. 256.]
    #  [  0.   0. 512. 512.] 重复三次

    # 改变先验框的尺度信息
    for i in range(len(ratios)):
        # 前三个anchor是1:1，中间三个是1:2，后面三个是2:1
        anchors[3 * i:3 * i + 3, 2] = anchors[3 * i:3 * i + 3, 2] * ratios[i][0]
        anchors[3 * i:3 * i + 3, 3] = anchors[3 * i:3 * i + 3, 3] * ratios[i][1]

    # 将anchors的原点从0移动到中心
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(share_layer_shape, anchors, stride=cfg.rpn_stride):
    """
    生成所有的先验框
    :param share_layer_shape: 共享特征层的shape
    :param anchors: 先验框的尺寸
    :param stride: 特征图对应到原图上的步长，也可以看作是感受野
    :return:
    """
    # 生成 8至600，步长为16 的一维矩阵，shape=(38,)
    coordinate_x = (np.arange(0, share_layer_shape[0], dtype=np.float32) + 0.5) * stride
    coordinate_y = (np.arange(0, share_layer_shape[1], dtype=np.float32) + 0.5) * stride

    # 生成高维矩阵,shape=(38, 38)
    coordinate_x, coordinate_y = np.meshgrid(coordinate_x, coordinate_y)

    # 将生成的二维矩阵展平,shape=(1444,)
    coordinate_x = np.reshape(coordinate_x, [-1])
    coordinate_y = np.reshape(coordinate_y, [-1])

    # 将它们按顺序堆叠,shape=(4,1444)
    coordinates = np.stack([
        coordinate_x,
        coordinate_y,
        coordinate_x,
        coordinate_y
    ], axis=0)

    # 转置,里面存储的是特征图上每个单元格的在实际图像中的坐标中心,shape=(1444,4)
    coordinates = np.transpose(coordinates)

    # 获取特征图单元格数目38*38=1444 以及 每个单元格上先验框的数量k=9
    number_of_anchors = np.shape(anchors)[0]
    k = np.shape(coordinates)[0]

    # 在anchors(9, 4)的0维度上添加一个维度(1, 9, 4) shifts的1维度上添加一个维度(1444, 1, 4) 相加得到框在原图中的坐标(1444, 9, 4)
    shifted_anchors = np.expand_dims(anchors, axis=0) + np.expand_dims(coordinates, axis=1)
    # reshape成(12996, 4)
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors


def get_anchors(share_layer_shape, image_shape):
    """
    生成先验框
    :param share_layer_shape: 共享特征层的shape
    :param image_shape: 原图的宽高
    :return: 网络的先验框
    """
    width, height = image_shape
    anchors = generate_anchors()
    network_anchors = shift(share_layer_shape, anchors)

    # 把先验框转换成小数的形式
    network_anchors[:, 0] = network_anchors[:, 0] / width
    network_anchors[:, 1] = network_anchors[:, 1] / height
    network_anchors[:, 2] = network_anchors[:, 2] / width
    network_anchors[:, 3] = network_anchors[:, 3] / height

    # 把大于1小于0的框裁剪到不越过边界
    network_anchors = np.clip(network_anchors, 0, 1)

    return network_anchors

