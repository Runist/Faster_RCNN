# -*- coding: utf-8 -*-
# @File : train.py
# @Author: Runist
# @Time : 2020-09-11 11:59
# @Software: PyCharm
# @Brief: 训练文件

from nets.backbone.ResNet import ResNet50
from nets import frcnn

from core import losses as losses_fn
from core.anchorGenerate import get_anchors
from core.dataReader import DataReader, get_classifier_train_data
from core.boxParse import BoundingBox

import config.config as cfg

from tensorflow.keras import optimizers, Input, models, utils
import tensorflow as tf
import numpy as np
import os


def write_to_log(summary_writer, step, **kwargs):
    """
    写入训练日志，
    :param summary_writer: summary对象
    :param step: 训练步数
    :param kwargs: 以字典形式传入
    :return:
    """
    with summary_writer.as_default():
        for key, value in kwargs.items():
            tf.summary.scalar(key, value, step=step)


def lr_schedule(e, mode='normal'):
    """
    学习率调整
    :param e: 当前训练的epoch
    :param mode: normal为每十个epoch学习率减半，cosine_anneal为余弦退火调整
    :return: rpn_lr, cls_lr
    """
    if mode == 'cosine_anneal':
        # 余弦退火调整学习率
        if e <= 4:
            rpn_lr = cfg.rpn_lr_max / 4 * e
            cls_lr = cfg.cls_lr_max / 4 * e
        else:
            rpn_lr = cfg.rpn_lr_max + 0.5 * (cfg.rpn_lr_max - cfg.rpn_lr_min) * (1 + np.cos(e / cfg.epoch * np.pi))
            cls_lr = cfg.cls_lr_max + 0.5 * (cfg.cls_lr_max - cfg.rpn_lr_min) * (1 + np.cos(e / cfg.epoch * np.pi))
        return rpn_lr, cls_lr
    else:
        if e % 10 == 0 and e != 0:
            cfg.lr /= 2

        return cfg.lr, cfg.lr


def main():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    img_input = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))

    share_layer = ResNet50(img_input)
    rpn = frcnn.rpn(share_layer, num_anchors=len(cfg.anchor_box_ratios) * len(cfg.anchor_box_scales))
    classifier = frcnn.classifier(share_layer, roi_input, cfg.num_rois, nb_classes=cfg.num_classes)

    model_rpn = models.Model(img_input, rpn)
    model_classifier = models.Model([img_input, roi_input], classifier)
    model_all = models.Model([img_input, roi_input], rpn + classifier)

    # model_rpn.compile(optimizer=optimizers.Adam(cfg.lr / 100),
    #                   loss={'regression': losses_fn.rpn_regr_loss(),
    #                         'classification': losses_fn.rpn_cls_loss()})
    #
    # model_classifier.compile(optimizer=optimizers.Adam(cfg.lr),
    #                          loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(cfg.num_classes - 1)],
    #                          metrics={'dense_class_{}'.format(cfg.num_classes): 'accuracy'})
    #
    # model_all.compile(optimizer=optimizers.SGD(0.001), loss='mae')

    # 生成38x38x9个先验框
    anchors = get_anchors(cfg.share_layer_shape, cfg.input_shape)

    # 根据先验框解析真实框
    box_parse = BoundingBox(anchors, max_threshold=cfg.rpn_max_overlap, min_threshold=cfg.rpn_min_overlap)

    reader = DataReader(cfg.annotation_path, cfg.input_shape, cfg.batch_size, box_parse)
    train = reader.read_data_and_split_data(cfg.valid_rate)
    train_step = len(train)

    train_dataset = iter(reader.make_datasets(train))

    # loss相关
    losses = np.zeros((train_step, 5))
    best_loss = np.Inf

    # 创建summary
    summary_writer = tf.summary.create_file_writer(logdir=cfg.summary_path)

    for e in range(1, cfg.epoch + 1):
        rpn_lr, cls_lr = lr_schedule(e, mode='cosine_anneal')
        print("Learning rate adjustment, rpn_lr: {}, cls_lr: {}".format(rpn_lr, cls_lr))

        model_rpn.compile(optimizer=optimizers.Adam(rpn_lr),
                          loss={'regression': losses_fn.rpn_regr_loss(),
                                'classification': losses_fn.rpn_cls_loss()})

        model_classifier.compile(optimizer=optimizers.Adam(cls_lr),
                                 loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(cfg.num_classes - 1)],
                                 metrics={'dense_class_{}'.format(cfg.num_classes): 'accuracy'})

        # keras可视化训练条
        progbar = utils.Progbar(train_step)
        print('Epoch {}/{}'.format(e, cfg.epoch))
        for i in range(train_step):

            # 读取数据
            image, classification, regression, bbox = next(train_dataset)

            # train_on_batch输出结果分成两种，一种只返回loss，第二种返回loss+metrcis，主要由model.compile决定
            # model_rpn单输出模型，且只有loss，没有metrics, 此时 return 为一个标量，代表这个 mini-batch 的 loss
            # 这里的loss_rpn返回一个列表，loss_rpn[0] = loss_rpn[1] + loss_rpn[2]
            loss_rpn = model_rpn.train_on_batch(image, [classification, regression])
            predict_rpn = model_rpn.predict_on_batch(image)

            # 将预测结果进行解码
            predict_boxes = box_parse.detection_out(predict_rpn, confidence_threshold=0)
            height, width, _ = np.shape(image[0])
            x_roi, y_class_label, y_classifier = get_classifier_train_data(predict_boxes,
                                                                           bbox[0],
                                                                           width,
                                                                           height,
                                                                           cfg.num_classes)

            if x_roi is None:
                continue

            # 平衡classifier的数据正负样本
            neg_samples = np.where(y_class_label[0, :, -1] == 1)
            pos_samples = np.where(y_class_label[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            if len(neg_samples) == 0:
                continue

            # 平衡正样本数量
            if len(pos_samples) < cfg.num_rois // 2:
                selected_pos_samples = pos_samples.tolist()
            else:
                # replace选取后是否放回
                selected_pos_samples = np.random.choice(pos_samples, cfg.num_rois//2, replace=False).tolist()

            # 平衡负样本数量
            if len(neg_samples) >= cfg.num_rois - len(selected_pos_samples):
                selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples), replace=False).tolist()
            else:
                selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples), replace=True).tolist()

            selected_samples = selected_pos_samples + selected_neg_samples

            # 训练分类器
            loss_class = model_classifier.train_on_batch([image,
                                                          x_roi[:, selected_samples, :]],
                                                         [y_class_label[:, selected_samples, :], y_classifier[:, selected_samples, :]])

            losses[i, 0] = loss_rpn[1]
            losses[i, 1] = loss_rpn[2]
            losses[i, 2] = loss_class[1]
            losses[i, 3] = loss_class[2]
            losses[i, 4] = loss_class[3]

            # 输出训练过程
            progbar.add(cfg.batch_size, [('rpn_cls', np.mean(losses[:i+1, 0])),
                                         ('rpn_regr', np.mean(losses[:i+1, 1])),
                                         ('detector_cls', np.mean(losses[:i+1, 2])),
                                         ('detector_regr', np.mean(losses[:i+1, 3]))])

        # 当一个epoch训练完了以后，输出训练指标
        else:
            loss_rpn_cls = np.mean(losses[:, 0])
            loss_rpn_regr = np.mean(losses[:, 1])
            loss_class_cls = np.mean(losses[:, 2])
            loss_class_regr = np.mean(losses[:, 3])
            class_acc = np.mean(losses[:, 4])

            curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr

            print('\nClassifier accuracy for bounding boxes from RPN: {:.4f}'.format(class_acc))
            print('Loss RPN classifier: {:.4f}'.format(loss_rpn_cls))
            print('Loss RPN regression: {:.4f}'.format(loss_rpn_regr))
            print('Loss Detector classifier: {:.4f}'.format(loss_class_cls))
            print('Loss Detector regression: {:.4f}'.format(loss_class_regr))

            print('The best loss is {:.4f}. The current loss is {:.4f}.'.format(best_loss, curr_loss))
            if curr_loss < best_loss:
                best_loss = curr_loss

            print('Saving weights.\n')
            model_all.save_weights("./logs/model/faster_rcnn_{:.4f}.h5".format(curr_loss))

            write_to_log(summary_writer,
                         step=e,
                         mean_class_acc=class_acc,
                         mean_loss_rpn_cls=loss_rpn_cls,
                         mean_loss_rpn_regr=loss_rpn_regr,
                         mean_loss_class_cls=loss_class_cls,
                         mean_loss_class_regr=loss_class_regr)


if __name__ == '__main__':
    main()
