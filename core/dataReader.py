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
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2 as cv


class DataReader:
    """
    tf.data.Dataset高速读取数据，提高GPU利用率
    """

    def __init__(self, data_path, input_shape, batch_size, box_parse):
        """
        :param data_path: 图片-标签 对应关系的txt文本路径
        :param input_shape: 输入层的宽高信息
        :param box_parse: box解析类对象
        :param max_boxes: 一张图最大检测预测框数量
        """
        self.data_path = data_path
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_regions = cfg.num_regions
        self.box_parse = box_parse

    def read_data_and_split_data(self, valid_rate):
        """
        读取图片的路径信息，并按照比例分为训练集和测试集
        :param valid_rate: 分割比例
        :return:
        """
        with open(self.data_path, "r") as f:
            files = f.readlines()

        # valid_rate为0，全为训练集
        if valid_rate == 0:
            return files

        split = int(valid_rate * len(files))
        train = files[split:]
        valid = files[:split]

        return train, valid

    @staticmethod
    def __rand(small=0., big=1.):
        return np.random.rand() * (big - small) + small

    def parse(self, annotation_line):
        """
        为tf.data.Dataset.map编写合适的解析函数，由于函数中某些操作不支持
        python类型的操作，所以需要用py_function转换，定义的格式如下
            Args:
              @param annotation_line: 是一行数据（图片路径 + 预测框位置）
        tf.py_function
            Args:
              第一个是要转换成tf格式的python函数，
              第二个输入的参数，
              第三个是输出的类型
        """

        if cfg.data_pretreatment == "random":
            image, bbox = tf.py_function(self._get_random_data, [annotation_line], [tf.float32, tf.float32])
        else:
            image, bbox = tf.py_function(self._get_data, [annotation_line], [tf.float32, tf.float32])

        # py_function没有解析List的返回值，所以要拆包 再合起来传出去
        classification, regression = tf.py_function(self.process_true_bbox, [bbox], [tf.float32, tf.float32])

        return image, classification, regression, bbox

    def _get_data(self, annotation_line):
        """
        不对数据进行增强处理，只进行简单的尺度变换和填充处理
        :param annotation_line: 一行数据
        :return: image, box_data
        """
        line = str(annotation_line.numpy(), encoding="utf-8").split()
        image_path = line[0]
        bbox = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)

        image_height, image_width = tf.shape(image)[:2]
        input_height, input_width = self.input_shape

        image_height_f = tf.cast(image_height, tf.float32)
        image_width_f = tf.cast(image_width, tf.float32)
        input_height_f = tf.cast(input_height, tf.float32)
        input_width_f = tf.cast(input_width, tf.float32)

        scale = min(input_width_f / image_width_f, input_height_f / image_height_f)
        new_height = image_height_f * scale
        new_width = image_width_f * scale

        # 将图片按照固定长宽比进行缩放 空缺部分 padding
        dx_f = (input_width - new_width) / 2
        dy_f = (input_height - new_height) / 2
        dx = tf.cast(dx_f, tf.int32)
        dy = tf.cast(dy_f, tf.int32)

        # 其实这一块不是双三次线性插值resize导致像素点放大255倍，原因是：无论是cv还是plt在面对浮点数时，仅解释0-1完整比例
        image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BICUBIC)
        new_image = tf.image.pad_to_bounding_box(image, dy, dx, input_height, input_width)

        # 生成image.shape的大小的全1矩阵
        image_ones = tf.ones_like(image)
        image_ones_padded = tf.image.pad_to_bounding_box(image_ones, dy, dx, input_height, input_width)
        # 做个运算，白色区域变成0，填充0的区域变成1，再* 128，然后加上原图，就完成填充灰色的操作
        image = (1 - image_ones_padded) * 128 + new_image

        # 将图片归一化到0和1之间
        image /= 255.
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

        # 为填充过后的图片，矫正bbox坐标，如果没有bbox需要检测annotation文件
        if len(bbox) <= 0:
            raise Exception("{} doesn't have any bounding boxes.".format(image_path))

        # np.random.shuffle(bbox)
        bbox[:, [0, 2]] = bbox[:, [0, 2]] * scale + dx_f
        bbox[:, [1, 3]] = bbox[:, [1, 3]] * scale + dy_f
        box_data = np.array(bbox, dtype='float32')

        # 将bbox的坐标变0-1
        box_data[:, 0] = box_data[:, 0] / cfg.input_shape[1]
        box_data[:, 1] = box_data[:, 1] / cfg.input_shape[0]
        box_data[:, 2] = box_data[:, 2] / cfg.input_shape[1]
        box_data[:, 3] = box_data[:, 3] / cfg.input_shape[0]

        return image, box_data

    def _get_random_data(self, annotation_line):
        """
        数据增强（改变长宽比例、大小、亮度、对比度、颜色饱和度）
        :param annotation_line: 一行数据
        :return: image, box_data
        """
        line = str(annotation_line.numpy(), encoding="utf-8").split()
        image_path = line[0]
        bbox = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)

        image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
        input_width, input_height = self.input_shape
        flip = False

        # 随机左右翻转50%
        if self.__rand(0, 1) > 0.5:
            image = tf.image.random_flip_left_right(image, seed=1)
            flip = True
        # 改变亮度，max_delta必须是float且非负数
        image = tf.image.random_brightness(image, 0.2)
        # 对比度调节
        image = tf.image.random_contrast(image, 0.3, 2.0)
        # 色相调节
        image = tf.image.random_hue(image, 0.15)
        # 饱和度调节
        image = tf.image.random_saturation(image, 0.3, 2.0)

        # 对图像进行缩放并且进行长和宽的扭曲，改变图片的比例
        image_ratio = self.__rand(0.7, 1.3)
        # 随机生成缩放比例，缩小或者放大
        scale = self.__rand(0.5, 1.5)

        # 50%的比例改变width, 50%比例改变height
        if self.__rand(0, 1) > 0.5:
            new_height = int(scale * input_height)
            new_width = int(input_width * scale * image_ratio)
        else:
            new_width = int(scale * input_width)
            new_height = int(input_height * scale * image_ratio)

        # 这里不以scale作为判断条件是因为，尺度缩放的时候，即使尺度小于1，但图像的长宽比会导致宽比input_shape大
        # 会导致第二种条件，图像填充为黑色
        if new_height < input_height or new_width < input_width:
            new_width = input_width if new_width > input_width else new_width
            new_height = input_height if new_height > input_height else new_height

            # 将变换后的图像，转换为416x416的图像，其余部分用灰色值填充。
            # 将图片按照固定长宽比进行缩放 空缺部分 padding
            dx = tf.cast(self.__rand(0, (input_width - new_width)) / 2, tf.int32)
            dy = tf.cast(self.__rand(0, (input_height - new_height)) / 2, tf.int32)

            # 按照计算好的长宽进行resize
            image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BICUBIC)
            new_image = tf.image.pad_to_bounding_box(image, dy, dx, input_height, input_width)

            # 生成image.shape的大小的全1矩阵
            image_ones = tf.ones_like(image)
            image_ones_padded = tf.image.pad_to_bounding_box(image_ones, dy, dx, input_height, input_width)
            # 做个运算，白色区域变成0，填充0的区域变成1，再* 128，然后加上原图，就完成填充灰色的操作
            image = (1 - image_ones_padded) * 128 + new_image

        else:
            # 按照计算好的长宽进行resize，然后进行自动的裁剪
            image = tf.image.resize(image, [new_height, new_width], method=tf.image.ResizeMethod.BICUBIC)
            image = tf.image.resize_with_crop_or_pad(image, input_height, input_width)

        # 将图片归一化到0和1之间
        # image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = (image - np.mean(image)) / np.std(image)
        # image /= 255.
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

        img = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
        img = tf.image.encode_jpeg(img)
        tf.io.write_file("image.jpg", img)

        # 为填充过后的图片，矫正bbox坐标，如果没有bbox需要检测annotation文件
        if len(bbox) <= 0:
            raise Exception("{} doesn't have any bounding boxes.".format(image_path))

        dx = (input_width - new_width) // 2
        dy = (input_height - new_height) // 2

        bbox[:, [0, 2]] = bbox[:, [0, 2]] * new_width / image_width + dx
        bbox[:, [1, 3]] = bbox[:, [1, 3]] * new_height / image_height + dy
        if flip:
            bbox[:, [0, 2]] = input_width - bbox[:, [2, 0]]

        # 定义边界
        bbox[:, 0:2][bbox[:, 0:2] < 0] = 0
        bbox[:, 2][bbox[:, 2] > input_width] = input_width
        bbox[:, 3][bbox[:, 3] > input_height] = input_height

        # 计算新的长宽
        box_w = bbox[:, 2] - bbox[:, 0]
        box_h = bbox[:, 3] - bbox[:, 1]
        # 去除无效数据
        bbox = bbox[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        box_data = np.array(bbox, dtype='float32')

        # 将bbox的坐标变0-1
        box_data[:, 0] = box_data[:, 0] / cfg.input_shape[1]
        box_data[:, 1] = box_data[:, 1] / cfg.input_shape[0]
        box_data[:, 2] = box_data[:, 2] / cfg.input_shape[1]
        box_data[:, 3] = box_data[:, 3] / cfg.input_shape[0]

        return image, box_data

    def process_true_bbox(self, box_data):
        """
        对真实框处理，首先会建立一个13x13，26x26，52x52的特征层，具体的shape是
        [b, n, n, 3, 25]的特征层，也就意味着，一个特征层最多可以存放n^2个数据
        :param box_data: 实际框的数据
        :return: 处理好后的 y_true
        """
        boxes = self.box_parse.assign_boxes(box_data)

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
            # TODO 为什么regression不需要限制负例数量
            classification[val_index] = -1

        classification = np.reshape(classification, [-1, 1])
        regression = np.reshape(regression, [-1, 5])

        return classification, regression

    def make_datasets(self, annotation, mode="train"):
        """
        用tf.data的方式读取数据，以提高gpu使用率
        :param annotation: 数据行[image_path, [x,y,w,h,class ...]]
        :param mode: 训练集or验证集tf.data运行一次
        :return: 数据集
        """
        # 这是GPU读取方式
        dataset = tf.data.Dataset.from_tensor_slices(annotation)
        if mode == "train":
            # 如果使用mosaic数据增强的方式，要先将4个路径合成一条数据，先传入
            if cfg.data_pretreatment == "mosaic":
                dataset = dataset.repeat().batch(4)

            # map的作用就是根据定义的 函数，对整个数据集都进行这样的操作
            # 而不用自己写一个for循环，如：可以自己定义一个归一化操作，然后用.map方法都归一化
            dataset = dataset.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # 打乱数据，这里的shuffle的值越接近整个数据集的大小，越贴近概率分布
            # 但是电脑往往没有这么大的内存，所以适量就好
            dataset = dataset.repeat().batch(self.batch_size).shuffle(buffer_size=cfg.shuffle_size)
            # prefetch解耦了 数据产生的时间 和 数据消耗的时间
            # prefetch官方的说法是可以在gpu训练模型的同时提前预处理下一批数据
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        else:
            # 验证集数据不需要增强
            cfg.data_pretreatment = 'normal'
            dataset = dataset.map(self.parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.repeat().batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset


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


def get_classifier_train_data(predict_boxes, true_boxes, img_w, img_h, num_classes):
    """
    生成classifier_train_data的训练数据
    :param predict_boxes: 预测框
    :param true_boxes: 真实框
    :param img_w: 图片的宽
    :param img_h: 图片的高
    :param num_classes: 分类数 + 1
    :return: roi_pooling层的输入， label列表， 9种尺度的回归坐标和具体类别
    """

    bboxes = true_boxes[:, :4].numpy()

    gta = np.zeros((len(bboxes), 4))
    # 将原图下0-1范围的框 变换到在 共享特征层从（38，38）尺度下的框
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
    for i in range(predict_boxes.shape[0]):
        x1 = int(round(predict_boxes[i, 0] * img_w / cfg.rpn_stride))
        y1 = int(round(predict_boxes[i, 1] * img_h / cfg.rpn_stride))
        x2 = int(round(predict_boxes[i, 2] * img_w / cfg.rpn_stride))
        y2 = int(round(predict_boxes[i, 3] * img_h / cfg.rpn_stride))

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
                label = int(true_boxes[best_idx, -1])

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
        return None, None, None

    x_roi = np.array(x_roi)
    y_class_label = np.array(y_class_label)
    y_classifier = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1)

    return np.expand_dims(x_roi, axis=0), np.expand_dims(y_class_label, axis=0), np.expand_dims(y_classifier, axis=0)


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
