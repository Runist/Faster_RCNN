# -*- coding: utf-8 -*-
# @File : frcnn.py
# @Author: Runist
# @Time : 2020/9/7 20:38
# @Software: PyCharm
# @Brief: faster rcnn模型结构


from nets.backbone.ResNet import classifier_layers
from tensorflow.keras import layers
from nets.RoiPooling import RoiPooling


def rpn(share_layer, num_anchors=9):
    """
    RPN网络
    :param share_layer: 经过backbone-ResNet(可更换)处理的特征层
    :param num_anchors: 特征层上每个格点上的anchor数量
    :return: 两个特征层
        一个用于输出置信度，内部是否包含物体，通道数为k(二元交叉熵)或2k(交叉熵)
        第二个用于输出预测框的坐标，通道数为4k

    """
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                      kernel_initializer='normal', name='rpn_conv1')(share_layer)

    x_class = layers.Conv2D(num_anchors, kernel_size=1, activation='sigmoid',
                            kernel_initializer='uniform', name='rpn_class')(x)

    x_regr = layers.Conv2D(num_anchors * 4, kernel_size=1,
                           kernel_initializer='zero', name='rpn_regress')(x)

    x_class = layers.Reshape((-1, 1), name="classification")(x_class)
    x_regr = layers.Reshape((-1, 4), name="regression")(x_regr)

    return [x_class, x_regr, share_layer]


def classifier(share_layer, input_rois, num_rois, nb_classes=21):
    """
    构建分类器
    :param share_layer: 经过backbone-ResNet(可更换)处理的共享特征层
    :param input_rois: roi input 的 Tensor
    :param num_rois: 输入网络中的候选框数量
    :param nb_classes: 20分类数量+1背景
    :return:
    """
    pooling_size = 14

    out_roi_pool = RoiPooling(pooling_size, num_rois)([share_layer, input_rois])

    out = classifier_layers(out_roi_pool)
    out = layers.TimeDistributed(layers.Flatten(), name="flatten")(out)

    out_class = layers.TimeDistributed(layers.Dense(nb_classes,
                                                    activation='softmax',
                                                    kernel_initializer='zero'),
                                       name='dense_class_{}'.format(nb_classes))(out)

    out_regr = layers.TimeDistributed(layers.Dense(4 * (nb_classes-1),
                                                   activation='linear',
                                                   kernel_initializer='zero'),
                                      name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]

