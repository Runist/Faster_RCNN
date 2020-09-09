# -*- coding: utf-8 -*-
# @File : generate_anchor.py
# @Author: Runist
# @Time : 2020/9/9 20:53
# @Software: PyCharm
# @Brief: 生成先验框

import numpy as np
# import tensorflow as tf

anchor_box_scales = [128, 256, 512]
anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]


def generate_anchors(sizes=None, ratios=None):

    if ratios is None:
        ratios = anchor_box_ratios
    if sizes is None:
        sizes = anchor_box_scales

    num_anchors = len(sizes) * len(ratios)

    # 每个点9个先验框，4个坐标
    anchors = np.zeros((num_anchors, 4))

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


def shift(shape, anchors, stride=cfg.rpn_stride):
    shift_x = (np.arange(0, shape[0], dtype=keras.backend.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[1], dtype=keras.backend.floatx()) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])

    shifts = np.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]

    k = np.shape(shifts)[0]

    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]),
                                                                                keras.backend.floatx())
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    return shifted_anchors


def get_anchors(shape, width, height):
    anchors = generate_anchors()
    network_anchors = shift(shape, anchors)
    network_anchors[:, 0] = network_anchors[:, 0] / width
    network_anchors[:, 1] = network_anchors[:, 1] / height
    network_anchors[:, 2] = network_anchors[:, 2] / width
    network_anchors[:, 3] = network_anchors[:, 3] / height
    network_anchors = np.clip(network_anchors, 0, 1)
    return network_anchors


if __name__ == '__main__':
    anchors = generate_anchors()
