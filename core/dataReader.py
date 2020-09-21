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

    @staticmethod
    def merge_bboxes(bboxes, cutx, cuty):
        """
        四张图的box的合并，合并前是都是基于0坐标的Box。现在要将box合并到同一个坐标系下
        :param bboxes:
        :param cutx:
        :param cuty:
        :return:
        """
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    # 如果左上角的坐标比分界线大，就不要了
                    if y1 > cuty or x1 > cutx:
                        continue
                    # 分界线在y1和y2之间。就取cuty
                    if y2 >= cuty >= y1:
                        y2 = cuty
                        # 类似于这样的宽或高太短的就不要了
                        if y2 - y1 < 5:
                            continue
                    if x2 >= cutx >= x1:
                        x2 = cutx
                        if x2 - x1 < 5:
                            continue

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue

                    if y2 >= cuty >= y1:
                        y1 = cuty
                        if y2 - y1 < 5:
                            continue

                    if x2 >= cutx >= x1:
                        x2 = cutx
                        if x2 - x1 < 5:
                            continue

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue

                    if y2 >= cuty >= y1:
                        y1 = cuty
                        if y2 - y1 < 5:
                            continue

                    if x2 >= cutx >= x1:
                        x1 = cutx
                        if x2 - x1 < 5:
                            continue

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue

                    if y2 >= cuty >= y1:
                        y2 = cuty
                        if y2 - y1 < 5:
                            continue

                    if x2 >= cutx >= x1:
                        x1 = cutx
                        if x2 - x1 < 5:
                            continue

                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)

        return merge_bbox

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

        if cfg.data_pretreatment == "mosaic":
            image, bbox = tf.py_function(self._get_random_data_with_mosaic, [annotation_line], [tf.float32, tf.float32])
        elif cfg.data_pretreatment == "random":
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
        box_data[:, 0] = box_data[:, 0] / image_width
        box_data[:, 1] = box_data[:, 1] / image_height
        box_data[:, 2] = box_data[:, 2] / image_width
        box_data[:, 3] = box_data[:, 3] / image_height

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
        input_height, input_width = self.input_shape
        flip = False

        # 随机左右翻转50%
        if self.__rand(0, 1) > 0.5:
            image = tf.image.random_flip_left_right(image, seed=1)
            flip = True
        # 改变亮度，max_delta必须是float且非负数
        image = tf.image.random_brightness(image, 0.2)
        # 对比度调节
        image = tf.image.random_contrast(image, 0.3, 2.0)
        # # 色相调节
        image = tf.image.random_hue(image, 0.15)
        # 饱和度调节
        image = tf.image.random_saturation(image, 0.3, 2.0)

        # 对图像进行缩放并且进行长和宽的扭曲，改变图片的比例
        image_ratio = self.__rand(0.6, 1.4)
        # 随机生成缩放比例，缩小或者放大
        scale = self.__rand(0.3, 1.5)

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
            dx = tf.cast((input_width - new_width) / 2, tf.int32)
            dy = tf.cast((input_height - new_height) / 2, tf.int32)

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
        image /= 255.
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

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

        # TODO 检查宽高的值到底是形变后的还是原图的
        # 将bbox的坐标变0-1
        box_data[:, 0] = box_data[:, 0] / image_width
        box_data[:, 1] = box_data[:, 1] / image_height
        box_data[:, 2] = box_data[:, 2] / image_width
        box_data[:, 3] = box_data[:, 3] / image_height

        return image, box_data

    def _get_random_data_with_mosaic(self, annotation_line, hue=.1, sat=1.5, val=1.5):
        """
        mosaic数据增强方式
        :param annotation_line: 4行图像信息数据
        :param hue: 色域变换的h色调
        :param sat: 饱和度S
        :param val: 明度V
        :return:
        """
        input_height, input_width = self.input_shape

        min_offset_x = 0.45
        min_offset_y = 0.45
        scale_low = 1 - min(min_offset_x, min_offset_y)
        scale_high = scale_low + 0.2

        image_datas = []
        box_datas = []

        # 定义分界线，用列表存储
        place_x = [0, 0, int(input_width * min_offset_x), int(input_width * min_offset_x)]
        place_y = [0, int(input_height * min_offset_y), int(input_width * min_offset_y), 0]
        for i in range(4):
            line = str(annotation_line[i].numpy(), encoding="utf-8").split()
            image_path = line[0]
            bbox = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

            # 打开图片
            image = Image.open(image_path)
            image = image.convert("RGB")
            # 图片的大小
            image_width, image_height = image.size

            # 是否翻转图片
            flip = self.__rand() < 0.5
            if flip and len(bbox) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                bbox[:, [0, 2]] = image_width - bbox[:, [2, 0]]

            # 对输入进来的图片进行缩放
            scale = self.__rand(scale_low, scale_high)
            new_height = int(scale * image_height)
            new_width = int(scale * image_width)
            image = image.resize((new_width, new_height), Image.BICUBIC)

            # 进行色域变换，hsv直接从色调、饱和度、明亮度上变化
            hue = self.__rand(-hue, hue)
            sat = self.__rand(1, sat) if self.__rand() < .5 else 1 / self.__rand(1, sat)
            val = self.__rand(1, val) if self.__rand() < .5 else 1 / self.__rand(1, val)
            x = rgb_to_hsv(np.array(image) / 255.)

            # 第一个通道是h
            x[..., 0] += hue
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            # 第二个通道是s
            x[..., 1] *= sat
            # 第三个通道是s
            x[..., 2] *= val
            x[x > 1] = 1
            x[x < 0] = 0
            image = hsv_to_rgb(x)

            image = Image.fromarray((image * 255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[i]
            dy = place_y[i]

            mosaic_image = Image.new('RGB', (input_width, input_height), (128, 128, 128))
            mosaic_image.paste(image, (dx, dy))
            mosaic_image = np.array(mosaic_image) / 255

            # 如果没有bbox需要检测annotation文件
            if len(bbox) <= 0:
                raise Exception("{} doesn't have any bounding boxes.".format(image_path))

            # 对box进行重新处理
            np.random.shuffle(bbox)
            # 重新计算bbox的宽高 乘上尺度 加上偏移
            bbox[:, [0, 2]] = bbox[:, [0, 2]] * new_width / image_width + dx
            bbox[:, [1, 3]] = bbox[:, [1, 3]] * new_height / image_height + dy

            # 定义边界(bbox[:, 0:2] < 0的到的是Bool型的列表，True值置为边界)
            bbox[:, 0:2][bbox[:, 0:2] < 0] = 0
            bbox[:, 2][bbox[:, 2] > input_width] = input_width
            bbox[:, 3][bbox[:, 3] > input_height] = input_height

            # 计算新的长宽
            bbox_w = bbox[:, 2] - bbox[:, 0]
            bbox_h = bbox[:, 3] - bbox[:, 1]

            # 去除无效数据
            bbox = bbox[np.logical_and(bbox_w > 1, bbox_h > 1)]
            bbox = np.array(bbox, dtype=np.float)

            image_datas.append(mosaic_image)
            box_datas.append(bbox)

        # 随机选取分界线，将图片放上去
        cutx = np.random.randint(int(input_width * min_offset_x), int(input_width * (1 - min_offset_x)))
        cuty = np.random.randint(int(input_height * min_offset_y), int(input_height * (1 - min_offset_y)))

        mosaic_image = np.zeros([input_height, input_width, 3])
        mosaic_image[:cuty, :cutx] = image_datas[0][:cuty, :cutx]
        mosaic_image[cuty:, :cutx] = image_datas[1][cuty:, :cutx]
        mosaic_image[cuty:, cutx:] = image_datas[2][cuty:, cutx:]
        mosaic_image[:cuty, cutx:] = image_datas[3][:cuty, cutx:]

        # 对框进行坐标系的处理
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        # 将box进行调整
        box_data = np.array(new_boxes, 'float32')

        return mosaic_image, box_data

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


def get_classifier_train_data(predict_boxes, true_boxes, num_classes):
    """
    生成classifier_train_data的训练数据
    :param predict_boxes: 预测框
    :param true_boxes: 真实框
    :param num_classes: 分类数 + 1
    :return: roi_pooling层的输入， label列表， 9种尺度的回归坐标和具体类别
    """
    width, height = cfg.share_layer_shape[:2]

    bboxes = true_boxes[:, :4].numpy()

    gta = np.zeros((len(bboxes), 4))
    # 将原图下0-1范围的框 变换到在 共享特征层从（38，38）尺度下的框
    for bbox_num, bbox in enumerate(bboxes):

        gta[bbox_num, 0] = int(round(bbox[0] * width))
        gta[bbox_num, 1] = int(round(bbox[1] * height))
        gta[bbox_num, 2] = int(round(bbox[2] * width))
        gta[bbox_num, 3] = int(round(bbox[3] * height))

    x_roi = []
    y_class_label = []
    y_class_regr_coords = []
    y_class_regr_label = []

    # 遍历每个预测框
    for i in range(predict_boxes.shape[0]):
        x1 = int(round(predict_boxes[i, 0] * width))
        y1 = int(round(predict_boxes[i, 1] * height))
        x2 = int(round(predict_boxes[i, 2] * width))
        y2 = int(round(predict_boxes[i, 3] * height))

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
            coords[label_pos: 4+label_pos] = [sx*tx, sy*ty, sw*tw, sh*th]
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


