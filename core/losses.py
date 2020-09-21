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
        # y_true [batch_size, num_anchor, num_classes+1]
        # y_pred [batch_size, num_anchor, num_classes]
        labels = y_true
        anchor_state = y_true[:, :, -1]  # -1 是需要忽略的, 0 是背景, 1 是存在目标
        classification = y_pred

        # 找出存在目标的先验框 有目标是1
        indices_for_object = tf.where(backend.equal(anchor_state, 1))   # 如果x,y都为None，就返回condition的坐标
        labels_for_object = tf.gather_nd(labels, indices_for_object)    # 从labels选indices_for_object个元素
        classification_for_object = tf.gather_nd(classification, indices_for_object)

        cls_loss_for_object = backend.binary_crossentropy(labels_for_object, classification_for_object)

        # 找出实际上为背景的先验框 没有目标是0
        indices_for_back = tf.where(backend.equal(anchor_state, 0))
        labels_for_back = tf.gather_nd(labels, indices_for_back)
        classification_for_back = tf.gather_nd(classification, indices_for_back)

        # 计算每一个先验框应该有的权重
        cls_loss_for_back = backend.binary_crossentropy(labels_for_back, classification_for_back)

        # 标准化，实际上是正样本的数量
        normalizer_pos = tf.where(backend.equal(anchor_state, 1))
        normalizer_pos = backend.cast(backend.shape(normalizer_pos)[0], 'float32')
        normalizer_pos = backend.maximum(backend.cast_to_floatx(1.0), normalizer_pos)

        normalizer_neg = tf.where(backend.equal(anchor_state, 0))
        normalizer_neg = backend.cast(backend.shape(normalizer_neg)[0], 'float32')
        normalizer_neg = backend.maximum(backend.cast_to_floatx(1.0), normalizer_neg)

        # 将所获得的loss除上正样本的数量
        cls_loss_for_object = backend.sum(cls_loss_for_object) / normalizer_pos
        cls_loss_for_back = ratio * backend.sum(cls_loss_for_back) / normalizer_neg

        # 总的loss
        loss = cls_loss_for_object + cls_loss_for_back

        return loss

    return cls_loss


def rpn_regr_loss(sigma=1.0):
    sigma_squared = sigma ** 2

    def smooth_l1(y_true, y_pred):
        # y_true [batch_size, num_anchor, 4+1]
        # y_pred [batch_size, num_anchor, 4]
        regression = y_pred
        regression_target = y_true[:, :, :-1]  # 最后一个维度，不能取到-1
        anchor_state = y_true[:, :, -1]

        # 找到正样本
        indices = tf.where(backend.equal(anchor_state, 1))                  # 如果x,y都为None，就返回condition的坐标
        regression = tf.gather_nd(regression, indices)                      # 从regression选indices个元素
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算 smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = backend.abs(regression - regression_target)
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
        x = y_true[:, :, 4 * num_classes:] - y_pred
        x_abs = backend.abs(x)
        x_bool = backend.cast(backend.less_equal(x_abs, 1.0), 'float32')
        loss = 4 * backend.sum(
            y_true[:, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / backend.sum(
            epsilon + y_true[:, :, :4 * num_classes])
        return loss

    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    return backend.mean(losses.categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
