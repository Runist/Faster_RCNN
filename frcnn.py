# -*- coding: utf-8 -*-
# @File : frcnn.py
# @Author: Runist
# @Time : 2020/9/7 20:38
# @Software: PyCharm
# @Brief:


from backbone.ResNet import ResNet50
from tensorflow.keras import layers, models


def get_rpn(base_layers, num_anchors=9):
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                      kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = layers.Conv2D(num_anchors, kernel_size=1, activation='sigmoid',
                            kernel_initializer='uniform', name='rpn_out_class')(x)

    x_regr = layers.Conv2D(num_anchors * 4, kernel_size=1,
                           kernel_initializer='zero', name='rpn_out_regress')(x)

    x_class = layers.Reshape((-1, 1), name="classification")(x_class)
    x_regr = layers.Reshape((-1, 4), name="regression")(x_regr)

    return [x_class, x_regr, base_layers]


def get_classifier(base_layers, input_rois, num_rois, nb_classes=21, trainable=False):
    pooling_regions = 14
    input_shape = (num_rois, 14, 14, 1024)
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]


def get_model(num_classes):
    inputs = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))

    base_layers = ResNet50(inputs)

    # num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn = get_rpn(base_layers, num_anchors=9)
    model_rpn = Model(inputs, rpn[:2])

    classifier = get_classifier(base_layers, roi_input, cfg.num_rois, nb_classes=num_classes, trainable=True)
    model_classifier = Model([inputs, roi_input], classifier)

    model = models.Model([inputs, roi_input], rpn[:2]+classifier)

    return model_rpn, model_classifier, model
