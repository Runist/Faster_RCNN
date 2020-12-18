# -*- coding: utf-8 -*-
# @File : losses.py
# @Author: Runist
# @Time : 2020-09-11 10:13
# @Software: PyCharm
# @Brief: 模型loss函数


from tensorflow.keras import backend, losses
import tensorflow as tf
import numpy as np


def rpn_cls_loss(ratio=3):
    def cls_loss(y_true, y_pred):
        """
        计算rpn 是否有物体的loss，可以直接用交叉熵计算损失，但这里计算
        :param y_true: 真实值 [batch_size, num_anchor, 1]
        :param y_pred: 预测值 [batch_size, num_anchor, 1]
        :return: rpn cls_loss
        """

        label_true = y_true[:, :, -1]       # 取出真实值的label rpn的label 1是物体，0是背景，-1是需要忽略的
        label_pred = y_pred

        # 找出存在目标的先验框 有目标是1
        indices_for_object = tf.where(backend.equal(label_true, 1))     # 如果x,y都为None，就返回condition的坐标
        labels_for_object = tf.gather_nd(y_true, indices_for_object)    # 根据indices选取真实值标签
        classification_for_object = tf.gather_nd(label_pred, indices_for_object)    # 选取预测值标签

        cls_loss_for_object = backend.binary_crossentropy(labels_for_object, classification_for_object)

        # 找出实际上为背景的先验框 没有目标是0
        indices_for_back = tf.where(backend.equal(label_true, 0))
        labels_for_back = tf.gather_nd(y_true, indices_for_back)
        classification_for_back = tf.gather_nd(label_pred, indices_for_back)

        # 计算每一个先验框应该有的权重
        cls_loss_for_back = backend.binary_crossentropy(labels_for_back, classification_for_back)

        # 标准化，计算是正样本的数量
        normalizer_pos = tf.where(backend.equal(label_true, 1))
        normalizer_pos = backend.cast(backend.shape(normalizer_pos)[0], 'float32')
        normalizer_pos = backend.maximum(backend.cast_to_floatx(1.0), normalizer_pos)

        # 计算负样本的数量
        normalizer_neg = tf.where(backend.equal(label_true, 0))
        normalizer_neg = backend.cast(backend.shape(normalizer_neg)[0], 'float32')
        normalizer_neg = backend.maximum(backend.cast_to_floatx(1.0), normalizer_neg)

        # 将所获得的loss除上样本的数量
        cls_loss_for_object = backend.sum(cls_loss_for_object) / normalizer_pos         # 物体的loss
        cls_loss_for_back = ratio * backend.sum(cls_loss_for_back) / normalizer_neg     # 背景的loss

        # 总的loss
        loss = cls_loss_for_object + cls_loss_for_back

        return loss

    return cls_loss


def rpn_regr_loss(sigma=1.0):
    sigma_squared = sigma ** 2

    def smooth_l1(y_true, y_pred):
        """
        计算rpn 建议框坐标的loss
        使用smooth l1 loss
        f(x) =  0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
                |x| - 0.5 / sigma^2          otherwise
        :param sigma: 是平滑参数，控制平滑区域
        :param y_true: 真实值 [batch_size, num_anchor, 4+1]
        :param y_pred: 预测值 [batch_size, num_anchor, 4]
        :return: rpn regr_loss
        """
        regression_pred = y_pred
        regression_true = y_true[:, :, :-1]   # 取rpn上的坐标
        label_true = y_true[:, :, -1]         # 取框内是否有物体的预测值

        # 找到只有物体的框，不要背景
        indices = tf.where(backend.equal(label_true, 1))                    # 如果x,y都为None，就返回condition的坐标
        regression_pred = tf.gather_nd(regression_pred, indices)            # 根据有物体的索引，取出预测框的相关坐标
        regression_true = tf.gather_nd(regression_true, indices)            # 取出真实框的坐标

        # 计算 smooth L1 loss
        regression_diff = backend.abs(regression_pred - regression_true)
        regression_loss = tf.where(  # tf.where用做判断条件
            backend.less(regression_diff, 1.0 / sigma_squared),             # 绝对值是否小于1
            0.5 * sigma_squared * backend.pow(regression_diff, 2),          # 如果是
            regression_diff - 0.5 / sigma_squared                           # 如果不是
        )

        # 除于N_cls
        normalizer = backend.maximum(1, backend.shape(indices)[0])
        normalizer = backend.cast(normalizer, dtype='float32')
        loss = backend.sum(regression_loss) / normalizer

        return loss

    return smooth_l1


def class_loss_regr(num_classes):
    epsilon = 1e-4

    def class_loss_regr_fixed_num(y_true, y_pred):
        """
        计算classifier的回归损失
        :param y_true: 真实值 [batch_size, num_rois, num_classes * 8]
        :param y_pred: 预测值 [batch_size, num_rois, num_classes * 4]
        :return: classifier regr_loss
        """
        regr_loss = 0
        batch_size = len(y_true)
        for i in range(batch_size):
            x = y_true[i, :, 4 * num_classes:] - y_pred[i, :, :]                    # 取出y_true后一半的数据，与y_pred做差值
            x_abs = backend.abs(x)                                                  # 计算绝对值
            x_bool = backend.cast(backend.less_equal(x_abs, 1.0), 'float32')        # 小于1的值

            # 1、差值绝对值小于1时0.5 * X^2，大于1的绝对值减0.5然后相加
            # 2、在乘上是否要计算这个loss
            # 3、求和在除以个数，得均值
            loss = 4 * backend.sum(
                y_true[i, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / backend.sum(
                epsilon + y_true[i, :, :4 * num_classes])
            regr_loss += loss

        return regr_loss / backend.constant(batch_size)

    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    """
    计算具体的分类loss
    :param y_true: 真实值 [batch_size, num_rois, 4+1]
    :param y_pred: 预测值 [batch_size, num_rois, 4+1]
    :return: classifier class_loss
    """
    return backend.mean(losses.categorical_crossentropy(y_true, y_pred))
