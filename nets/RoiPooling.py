# -*- coding: utf-8 -*-
# @File : RoiPooling.py
# @Author: Runist
# @Time : 2020/9/8 20:19
# @Software: PyCharm
# @Brief:

from tensorflow.keras import layers, backend
import tensorflow as tf


class RoiPooling(layers.Layer):
    def __init__(self, pool_size, num_rois, **kwargs):
        """
        RoiPooling层
        :param pool_size: 最后resize成 pool_size * pool_size的大小
        :param num_rois: 输入网络中的候选框数量
        :param kwargs:
        """
        self.pool_size = pool_size
        self.num_rois = num_rois
        super(RoiPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        """
        计算输出的shape
        :param input_shape: 输入的shape
        :return:
        """
        return None, self.num_rois, self.pool_size, self.pool_size, self.channels

    def call(self, inputs, mask=None):

        assert(len(inputs) == 2)

        # 共享特征层
        feature_map = inputs[0]
        # 经过网络处理的候选框
        rois = inputs[1]

        outputs = []

        # 遍历每一个候选框（一般每次输入32个）
        for roi_idx in range(self.num_rois):

            # 将坐标从输入中取出来
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = backend.cast(x, 'int32')
            y = backend.cast(y, 'int32')
            w = backend.cast(w, 'int32')
            h = backend.cast(h, 'int32')

            # 将输入的特征层看作是图像，截取候选框区域的图像，然后resize成 pool_size * pool_size的大小
            rs = tf.image.resize(feature_map[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            # 将所有截取到的图像保存至一个列表
            outputs.append(rs)

        # 用concat合并所有截取图像
        final_output = backend.concatenate(outputs, axis=0)
        final_output = backend.reshape(final_output,
                                       (1, self.num_rois, self.pool_size, self.pool_size, self.channels))

        final_output = backend.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output

