# -*- coding: utf-8 -*-
# @File : ResNet.py
# @Author: Runist
# @Time : 2020/9/6 13:16
# @Software: PyCharm
# @Brief: ResNet 的 backbone

from tensorflow.keras import layers, models, applications


class BasicResBlock(layers.Layer):

    def __init__(self, filters, strides=1, **kwargs):
        """

        :param filters: 卷积核的数量
        :param strides: 为1时候不改变特征层宽高，为2就减半
        :param kwargs:
        """
        filter1, filter2, filter3 = filters
        super(BasicResBlock, self).__init__(**kwargs)

        self.conv1 = layers.Conv2D(filter1, kernel_size=1, strides=strides, use_bias=False)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filter2, kernel_size=3, strides=strides, use_bias=False, padding='same')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(filter3, kernel_size=1, strides=strides, use_bias=False)
        self.bn3 = layers.BatchNormalization()

        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False, **kwargs):
        """

        :param inputs: 输入Tensor
        :param training: 用在训练过程和预测过程中，控制其生效与否
        :param kwargs:
        :return:
        """
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([inputs, x])
        x = self.relu(x)

        return x


class BottleneckResBlock(layers.Layer):

    def __init__(self, filters, strides=1, **kwargs):
        """

        :param filters: 卷积核的数量
        :param strides: 为1时候不改变特征层宽高，为2就减半
        :param kwargs:
        """
        filter1, filter2, filter3 = filters
        super(BottleneckResBlock, self).__init__(**kwargs)

        self.shortcut = layers.Conv2D(filter3, kernel_size=1, strides=strides, use_bias=False)
        self.shortcut_bn = layers.BatchNormalization()

        self.conv1 = layers.Conv2D(filter1, kernel_size=1, strides=strides, use_bias=False)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filter2, kernel_size=3, use_bias=False, padding='same')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(filter3, kernel_size=1, use_bias=False)
        self.bn3 = layers.BatchNormalization()

        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False, **kwargs):
        """

        :param inputs: 输入Tensor
        :param training: 用在训练过程和预测过程中，控制其生效与否
        :param kwargs:
        :return:
        """
        x = self.shortcut(inputs)
        x = self.shortcut_bn(x, training=training)
        shortcut = self.relu(x)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([shortcut, x])
        x = self.relu(x)

        return x


def make_layer(filters, num, name, strides=2):
    """

    :param filters: 卷积核的数量
    :param num: 堆叠残差块的数量
    :param name: 这一层卷积的名字
    :param strides: 步长
    :return:
    """
    layer_list = [BottleneckResBlock(filters, strides=strides)]
    for _ in range(1, num):
        layer_list.append(BasicResBlock(filters))

    return models.Sequential(layer_list, name=name)


def ResNet50(input_image):

    # input_shape(None, 600, 600, 3)
    x = layers.ZeroPadding2D((3, 3))(input_image)

    # (606, 606, 3)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, name='conv1', use_bias=True)(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)

    # (300, 300, 64)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME')(x)

    # (150, 150, 64)
    x = make_layer([64, 64, 256], 3, 'conv_2x', strides=1)(x)

    # (150, 150, 256)
    x = make_layer([128, 128, 512], 4, 'conv_3x', strides=2)(x)

    # (75, 75, 512)
    feature_map = make_layer([256, 256, 1024], 6, 'conv_4x', strides=2)(x)

    # (38, 38, 512)
    return feature_map

