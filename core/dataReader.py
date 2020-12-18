# -*- coding: utf-8 -*-
# @File : dataReader.py
# @Author: Runist
# @Time : 2020/5/22 10:39
# @Software: PyCharm
# @Brief: 数据集读取


import tensorflow as tf
import numpy as np
import config.config as cfg
from PIL import Image
import cv2 as cv


class DataReader(object):
    def __init__(self, data_path, box_parse, batch_size, input_shape=cfg.input_shape):
        """
        :param data_path: 图片-标签 对应关系的txt文本路径
        :param box_parse: box解析类对象
        :param batch_size: 批处理数量
        :param input_shape: 输入层的宽高信息
        """
        self.data_path = data_path
        self.box_parse = box_parse
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_regions = cfg.num_regions
        self.train_lines = self.read_data_and_split_data()

    def read_data_and_split_data(self):
        """
        读取图片的路径信息，并按照比例分为训练集和测试集
        :return:
        """
        with open(self.data_path, "r", encoding='utf-8') as f:
            files = f.readlines()

        return files

    def get_random_data(self, annotation_line, hue=.1, sat=1.5, val=1.5):
        """
        数据增强（改变长宽比例、大小、亮度、对比度、颜色饱和度）
        :param annotation_line: 一行数据
        :param hue: 色调抖动
        :param sat: 饱和度抖动
        :param val: 明度抖动
        :return: image, box_data
        """
        line = annotation_line.split()
        image = Image.open(line[0])

        image_width, image_height = image.size
        input_width, input_height = self.input_shape

        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # 随机生成缩放比例，缩小或者放大
        scale = rand(0.5, 1.5)
        # 随机变换长宽比例
        new_ar = input_width / input_height * rand(0.7, 1.3)

        if new_ar < 1:
            new_height = int(scale * input_height)
            new_width = int(new_height * new_ar)
        else:
            new_width = int(scale * input_width)
            new_height = int(new_width / new_ar)

        image = image.resize((new_width, new_height), Image.BICUBIC)

        dx = rand(0, (input_width - new_width))
        dy = rand(0, (input_height - new_height))
        new_image = Image.new('RGB', (input_width, input_height), (128, 128, 128))
        new_image.paste(image, (int(dx), int(dy)))
        image = new_image

        # 翻转图片
        flip = rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 图像增强
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv.cvtColor(np.array(image, np.float32)/255, cv.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image = cv.cvtColor(x, cv.COLOR_HSV2RGB)*255

        # 为填充过后的图片，矫正box坐标，如果没有box需要检测annotation文件
        if len(box) <= 0:
            raise Exception("{} doesn't have any bounding boxes.".format(image_path))

        box[:, [0, 2]] = box[:, [0, 2]] * new_width / image_width + dx
        box[:, [1, 3]] = box[:, [1, 3]] * new_height / image_height + dy
        # 若翻转了图像，框也需要翻转
        if flip:
            box[:, [0, 2]] = input_width - box[:, [2, 0]]

        # 定义边界
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > input_width] = input_width
        box[:, 3][box[:, 3] > input_height] = input_height

        # 计算新的长宽
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        # 去除无效数据
        box = box[np.logical_and(box_w > 1, box_h > 1)]
        box_data = np.array(box, dtype='float32')

        # 将bbox的坐标变0-1
        box_data[:, 0] = box_data[:, 0] / input_width
        box_data[:, 1] = box_data[:, 1] / input_height
        box_data[:, 2] = box_data[:, 2] / input_width
        box_data[:, 3] = box_data[:, 3] / input_height

        return image, box_data

    def generate(self):
        """
        数据生成器
        :return: image, rpn训练标签， 真实框数据
        """

        i = 0
        n = len(self.train_lines)

        while True:
            image_data = []
            box_data = []
            classification_data = []
            regression_data = []

            if i == 0:
                np.random.shuffle(self.train_lines)

            j = 0
            while j < self.batch_size:
                image, bbox = self.get_random_data(self.train_lines[i])
                i = (i + 1) % n

                if len(bbox) == 0:
                    continue

                j += 1
                # 计算真实框对应的先验框，与这个先验框应当有的预测结果
                boxes = self.box_parse.assign_boxes(bbox)

                classification = boxes[:, 4]
                regression = boxes[:, :]

                mask_pos = classification[:] > 0
                num_pos = len(classification[mask_pos])

                if num_pos > self.num_regions / 2:
                    # 若正例超过128，则从多余的正例中随机采样置为-1
                    val_index = np.random.choice(np.where(mask_pos)[0].tolist(),
                                                 int(num_pos - self.num_regions / 2),
                                                 replace=False)
                    classification[val_index] = -1
                    regression[val_index, -1] = -1

                mask_neg = classification[:] == 0
                num_neg = len(classification[mask_neg])
                mask_pos = classification[:] > 0
                num_pos = len(classification[mask_pos])

                if num_neg + num_pos > self.num_regions:
                    # 若负例超过128，则把正例和负例降至256
                    val_index = np.random.choice(np.where(mask_neg)[0].tolist(),
                                                 int(num_neg + num_pos - self.num_regions),
                                                 replace=False)

                    classification[val_index] = -1

                classification = np.reshape(classification, [-1, 1])
                regression = np.reshape(regression, [-1, 5])

                image /= 255

                image_data.append(image)
                box_data.append(bbox)
                classification_data.append(classification)
                regression_data.append(regression)

            image_data = np.array(image_data)
            classification_data = np.array(classification_data, dtype=np.float32)
            regression_data = np.array(regression_data, dtype=np.float32)

            rpn_y = [classification_data, regression_data]

            yield image_data, rpn_y, box_data


def rand(small=0., big=1.):
    return np.random.rand() * (big - small) + small


def iou(box_a, box_b):
    """
    根据输入的两个框的坐标，计算iou，
    :param box_a: 第一个框的坐标
    :param box_b: 第二个框的坐标
    :return: iou
    """
    # 如果出现左上角的坐标大过右下角的坐标，则返回iou为0
    if box_a[0] >= box_a[2] or box_a[1] >= box_a[3] or box_b[0] >= box_b[2] or box_b[1] >= box_b[3]:
        return 0.0

    x = max(box_a[0], box_b[0])
    y = max(box_a[1], box_b[1])
    w = min(box_a[2], box_b[2]) - x
    h = min(box_a[3], box_b[3]) - y

    if w < 0 or h < 0:
        # 2个框不相交，分子为0，iou = 0
        return 0.0

    # 计算相交面积
    intersect_area = w * h
    # 计算并集面积
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = box_a_area + box_b_area - intersect_area

    return intersect_area / union_area


def get_classifier_train_data(predict_boxes, true_boxes, img_w, img_h, batch_size, num_classes):
    """
    生成classifier_train_data的训练数据
    :param predict_boxes: 预测框
    :param true_boxes: 真实框
    :param img_w: 图片的宽
    :param img_h: 图片的高
    :param batch_size: 批处理数量
    :param num_classes: 分类数 + 1
    :return: roi_pooling层的输入， label列表， 9种尺度的回归坐标和具体类别
    """

    batch_x_roi = []
    batch_y_class_label = []
    batch_y_classifier = []
    valid_roi = []    # 记录有效roi数据

    # 将原图下0-1范围的框 变换到在 共享特征层从（38，38）尺度下的框
    for b in range(batch_size):
        t_boxes = true_boxes[b]
        p_boxes = predict_boxes[b]

        bboxes = t_boxes[:, :4]
        gta = np.zeros((len(bboxes), 4))

        for bbox_num, bbox in enumerate(bboxes):

            gta[bbox_num, 0] = int(round(bbox[0] * img_w / cfg.rpn_stride))
            gta[bbox_num, 1] = int(round(bbox[1] * img_h / cfg.rpn_stride))
            gta[bbox_num, 2] = int(round(bbox[2] * img_w / cfg.rpn_stride))
            gta[bbox_num, 3] = int(round(bbox[3] * img_h / cfg.rpn_stride))

        x_roi = []
        y_class_label = []
        y_class_regr_coords = []
        y_class_regr_label = []

        # 遍历每个预测框
        for i in range(p_boxes.shape[0]):
            x1 = int(round(p_boxes[i, 0] * img_w / cfg.rpn_stride))
            y1 = int(round(p_boxes[i, 1] * img_h / cfg.rpn_stride))
            x2 = int(round(p_boxes[i, 2] * img_w / cfg.rpn_stride))
            y2 = int(round(p_boxes[i, 3] * img_h / cfg.rpn_stride))

            best_iou = 0.0
            best_idx = -1

            # 遍历每个真实框
            for bbox_num in range(len(bboxes)):
                curr_iou = iou(gta[bbox_num], [x1, y1, x2, y2])

                # 筛选出最优的iou和它的索引
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_idx = bbox_num

            # 如果这次筛选出来的best_iou小于0.1，说明是很好判断的背景，没必要训练
            if best_iou < cfg.classifier_min_overlap:
                continue
            else:
                w = x2 - x1
                h = y2 - y1
                x_roi.append([x1, y1, w, h])

                if cfg.classifier_min_overlap <= best_iou < cfg.classifier_max_overlap:
                    # iou在0.1~0.5之间的，认为是背景，可以训练
                    label = -1
                elif cfg.classifier_max_overlap <= best_iou:
                    # 获取框内物体分类索引（具体是什么东西）
                    label = int(t_boxes[best_idx, -1])

                    # 将共享特征层上的坐标用论文公式转换一下
                    cxg = (gta[best_idx, 0] + gta[best_idx, 2]) / 2.0
                    cyg = (gta[best_idx, 1] + gta[best_idx, 3]) / 2.0

                    # 左上角坐标变换为中心坐标
                    cx = x1 + w / 2.0
                    cy = y1 + h / 2.0

                    tx = (cxg - cx) / float(w)
                    ty = (cyg - cy) / float(h)
                    tw = np.log((gta[best_idx, 2] - gta[best_idx, 0]) / float(w))
                    th = np.log((gta[best_idx, 3] - gta[best_idx, 1]) / float(h))
                else:
                    print('roi = {}'.format(best_iou))
                    raise RuntimeError

            # 创建分类信息的列表
            class_label = num_classes * [0]
            class_label[label] = 1

            # 将符合条件的预测框的分类信息存储在一个二维列表中
            y_class_label.append(class_label)

            coords = [0.0] * 4 * (num_classes - 1)
            labels = [0.0] * 4 * (num_classes - 1)

            # 如果不是背景的话，才有必要记录边框坐标参数
            if label != -1:
                label_pos = 4 * label
                sx, sy, sw, sh = cfg.classifier_regr_std
                # 将回归参数×classifier_regr_std后存放到coords的对应位置上，将对应的labels四个位置全置1。
                coords[label_pos: 4+label_pos] = [sx * tx, sy * ty, sw * tw, sh * th]
                labels[label_pos: 4+label_pos] = [1, 1, 1, 1]
                y_class_regr_coords.append(coords)
                y_class_regr_label.append(labels)
            else:
                y_class_regr_coords.append(coords)
                y_class_regr_label.append(labels)

        # 如果x_roi为0，说明各个预测框与anchors的iou都很小，都是很简单的背景，则说明rpn没有训练好
        if len(x_roi) == 0:
            continue

        x_roi = np.array(x_roi)
        y_class_label = np.array(y_class_label)
        y_classifier = np.concatenate([np.array(y_class_regr_label, dtype=np.float32),
                                       np.array(y_class_regr_coords, dtype=np.float32)], axis=1)

        # 平衡classifier的数据正负样本
        neg_samples = np.where(y_class_label[:, -1] == 1)   # 背景的数量
        pos_samples = np.where(y_class_label[:, -1] == 0)   # 前景的数量

        if len(neg_samples) > 0:
            neg_samples = neg_samples[0]
        else:
            neg_samples = []

        if len(pos_samples) > 0:
            pos_samples = pos_samples[0]
        else:
            pos_samples = []

        # 若负样本数量为0
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
            selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
                                                    replace=False).tolist()
        else:
            selected_neg_samples = np.random.choice(neg_samples, cfg.num_rois - len(selected_pos_samples),
                                                    replace=True).tolist()

        selected_samples = selected_pos_samples + selected_neg_samples

        x_roi = x_roi[selected_samples, :]
        y_class_label = y_class_label[selected_samples, :]
        y_classifier = y_classifier[selected_samples, :]

        valid_roi.append(b)
        batch_x_roi.append(x_roi)
        batch_y_class_label.append(y_class_label)
        batch_y_classifier.append(y_classifier)

    batch_x_roi = np.array(batch_x_roi)
    batch_y_class_label = np.array(batch_y_class_label)
    batch_y_classifier = np.array(batch_y_classifier)

    return batch_x_roi, batch_y_class_label, batch_y_classifier, valid_roi


def get_new_image_size(width, height, short_side=600):
    """
    获得图像resize后宽高，依据短边来resize
    :param width: 图像原来的宽
    :param height: 图像原来的高
    :param short_side: 图像的短边
    :return: 新的宽高
    """
    scale = max(short_side / width, short_side / height)
    new_w = int(width * scale)
    new_h = int(height * scale)

    return new_w, new_h
