# -*- coding: utf-8 -*-
# @File : boxParse.py
# @Author: Runist
# @Time : 2020-09-14 10:40
# @Software: PyCharm
# @Brief: 预测框、真实框互相转换的文件
import numpy as np
import tensorflow as tf
from core.anchorGenerate import get_anchors


class BoundingBox(object):
    def __init__(self, anchors, max_threshold=0.7, min_threshold=0.3, nms_thresh=0.7, top_k=300):
        """
        预测框、先验框解析
        :param anchors: 先验框对象，如果没有，就直接按照特征层shape为38x38生成
        :param max_threshold: iou的上限阈值
        :param min_threshold: iou的下限阈值
        :param nms_thresh: nms的重叠阈值
        :param top_k: 前300个
        """
        self.anchors = anchors
        self.num_anchors = 0 if anchors is None else len(anchors)
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.nms_thresh = nms_thresh
        self.top_k = top_k

    def iou(self, box):
        """
        计算真实框与先验框的iou
        :param box: 真实框数据
        :return: iou
        """
        # 计算出每个真实框与所有的先验框的iou
        # 判断真实框与先验框的重合情况
        inter_leftup = np.maximum(self.anchors[:, :2], box[:2])
        inter_rightdown = np.minimum(self.anchors[:, 2:4], box[2:])

        inter_wh = inter_rightdown - inter_leftup
        # 防止先验框与真实框重叠部分的宽高为负数
        inter_wh = np.maximum(inter_wh, 0)

        # 重叠部分的面积
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 真实框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0]) * (self.anchors[:, 3] - self.anchors[:, 1])
        # 计算iou
        union = area_true + area_gt - inter

        iou = inter / union

        return iou

    def ignore_box(self, box):
        """
        计算无效先验框，iou在0.3-0.7之间的数据
        :param box: 真实框数据
        :return: shape=(len(anchors))，存储的是各个先验框与真实框的iou
        """
        iou = self.iou(box)

        ignored_box = np.zeros((self.num_anchors, 1))

        # 找到每一个真实框，按照阈值标准筛选iou0.3~0.7的框，这些框是没用的，不参与到Loss计算
        assign_mask = (iou > self.min_threshold) & (iou < self.max_threshold)

        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        # 至此assign_mask已经全是True or False的shape=（len(anchor),）的二维矩阵
        ignored_box[:, 0][assign_mask] = iou[assign_mask]

        # .ravel()把2维的矩阵降维成1维，np.squeeze .flatten()也能做到，但.ravel()速度更快，也可以做到
        # .flatten()不会修改到原始矩阵相当于copy一份出来
        return ignored_box.ravel()

    def encode_box(self, box):
        """
        计算真实框输入到网络的坐标和iou
        :param box: 真实框数据
        :return:
        """
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_anchors, 5))

        # 找到每一个iou大于0.7的框
        assign_mask = iou > self.max_threshold

        # 如果从上述条件中没有选出任何先验框，则选用iou最大的框
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        # 给最后一个维度标签位  置为iou
        encoded_box[:, -1][assign_mask] = iou[assign_mask]

        # 根据assign_mask，找到对应的先验框
        assigned_anchors = self.anchors[assign_mask]
        # 逆向编码，将真实框转化为FasterRCNN预测结果的格式
        # 先计算真实框的中心与长宽
        box_xy = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # 再计算重合度较高的先验框的中心与长宽
        anchors_xy = 0.5 * (assigned_anchors[:, :2] + assigned_anchors[:, 2:4])
        anchors_wh = (assigned_anchors[:, 2:4] - assigned_anchors[:, :2])

        # 逆向求取FasterRCNN应该有的预测结果 - 公式在论文中
        encoded_box[:, :2][assign_mask] = box_xy - anchors_xy
        encoded_box[:, :2][assign_mask] /= anchors_wh
        encoded_box[:, :2][assign_mask] *= 4

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / anchors_wh)
        encoded_box[:, 2:4][assign_mask] *= 4

        return encoded_box

    def decode_boxes(self, predictions, anchors):
        """
        对预测框进行解码，相当于encode_box的逆操作
        :param predictions: regression, shape=(12996, 4)
        :param anchors: 先验框对象
        :return: 解码后的boxes矩阵, shape=(12996, 4)
        """
        # 获得先验框的宽与高
        anchors_width = anchors[:, 2] - anchors[:, 0]
        anchors_height = anchors[:, 3] - anchors[:, 1]

        # 获得先验框的中心点
        anchors_center_x = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchors_center_y = 0.5 * (anchors[:, 3] + anchors[:, 1])

        # 计算预测框离先验框中心的偏移量——平移量（因为编码的时候乘*4，现在除掉）
        decode_bbox_center_x = predictions[:, 0] * anchors_width / 4
        decode_bbox_center_x += anchors_center_x
        decode_bbox_center_y = predictions[:, 1] * anchors_height / 4
        decode_bbox_center_y += anchors_center_y

        # 预测框的实际宽与高的求取——尺度变换
        decode_bbox_width = np.exp(predictions[:, 2] / 4)
        decode_bbox_width *= anchors_width
        decode_bbox_height = np.exp(predictions[:, 3] / 4)
        decode_bbox_height *= anchors_height

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)

        # 防止超出0与1
        decode_bbox = np.maximum(decode_bbox, 0.0)
        decode_bbox = np.minimum(decode_bbox, 1.0)

        return decode_bbox

    def assign_boxes(self, boxes):
        """
        根据真实框和先验框 计算转换后的框，并通过计算iou筛选无用框
        过assign_boxes我们就获得了，输入进来的这张图片，经过网络计算之后，输出的预测结果是什么样子的
        :param boxes: 真实框数据
        :return: 转换后的框数据
        """
        box_data = np.zeros((self.num_anchors, 4 + 1))

        # 先把标签全部置为0
        box_data[:, 4] = 0.0
        if len(boxes) == 0:
            return box_data

        # 对每一个真实框都进行iou计算
        # apply_along_axis作用: 将arr数组的每一个元素经过func函数变换形成的一个新数组
        # func是我们写的一个函数
        # axis表示函数func对arr是作用于行还是列
        # arr便是我们要进行操作的数组了
        ignored_boxes = np.apply_along_axis(self.ignore_box, 1, boxes[:, :4])
        # 取重合程度最大的先验框，并且获取这个先验框的index
        ignored_boxes = ignored_boxes.reshape(-1, self.num_anchors, 1)
        # 一张图片中多个框是分别计算iou的，但要把所有iou整合到一个anchors矩阵上，在0维度上比较最大值
        ignore_iou = ignored_boxes[:, :, 0].max(axis=0)

        ignore_iou_mask = ignore_iou > 0

        # 将需要忽略的先验框类别置为-1
        box_data[:, 4][ignore_iou_mask] = -1

        # (n, num_anchors, 5) n代表这个图中有n个坐标，5则是前4个是转换后的坐标，最后一个是这个框的iou
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)

        # 在同一个位置上，有可能会有多个物体的iou同时超过0.7
        # 由于apply_along_axis是在不同层上计算的，但最后只会输出一个先验框矩阵，
        # 为了解决一个位置上有多个框的问题，所以选出iou最大的值作为这个位置的框
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        # 这个argmax选出来的是0维度上的索引,shape=(12996,)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        # 筛选出iou大于0的先验框位置
        best_iou_mask = best_iou > 0

        # 根据iou掩膜确定有效先验框的索引
        best_iou_idx = best_iou_idx[best_iou_mask]

        # best_iou_idx - 0维度上的索引
        # best_iou_mask - 1维度上的索引

        # 将筛选出来的先验框坐标赋值到assignment，4为0代表为背景，为1代表是有物体
        box_data[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, best_iou_mask, :4]
        box_data[:, 4][best_iou_mask] = 1

        return box_data

    def detection_out(self, predictions, anchors, confidence_threshold=0.5):
        """
        将0-1的预测结果转换成在特征图上的长宽，并进行NMS处理
        :param predictions: rpn模型的预测
        :param anchors: 先验框对象
        :param confidence_threshold: 置信度阈值
        :return: 经过NMS处理后rpn上的坐标
        """
        p_classification = predictions[0]       # 是背景还是物体
        p_regression = predictions[1]           # 共享特征层上的坐标

        pred = []

        # 对每一个图片进行处理，regression 第一个维度是batch size大小，需要遍历为所有共享特征层输出结果
        for i in range(p_regression.shape[0]):
            # 对张特征图上的regression进行bbox的解码
            decode_bbox = self.decode_boxes(p_regression[i], anchors)

            # 取出置信度数据
            confidence = p_classification[i, :, 0]
            # 大于 confidence_threshold 则认为有物体
            mask = confidence > confidence_threshold

            # 取出得分高于confidence_threshold的框
            boxes_to_process = decode_bbox[mask]
            score_to_process = confidence[mask]

            # 非极大抑制，去掉box重合程度高的那一些框，top_k是指最多可以通过nms获得多少个框
            nms_index = tf.image.non_max_suppression(boxes_to_process, score_to_process, self.top_k,
                                                     iou_threshold=self.nms_thresh)

            # 取出在非极大抑制中效果较好的 框和得分
            good_boxes = tf.gather(boxes_to_process, nms_index).numpy()
            good_score = tf.gather(score_to_process, nms_index).numpy()
            good_score = np.expand_dims(good_score, axis=1)

            predict_boxes = np.concatenate((good_score, good_boxes), axis=1)
            argsort = np.argsort(predict_boxes[:, 0])[::-1]
            predict_boxes = predict_boxes[argsort]

            pred.append(predict_boxes[:, 1:])

        return pred
