import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
def f(x):
    n = len(x)  # 确定传入参数个数
    if n==2:
        return (x[1][1] - x[0][1]) / (x[1][0] - x[0][0])
    x0 = deepcopy(x)
    x1 = deepcopy(x)
    xk1 = x0.pop(n-2)
    xk = x1.pop()

    return (f(x0) - f(x1)) / (xk[0] - xk1[0])#! /usr/bin/env python#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : evaluate.py
#   Author      : YunYang1994
#   Created date: 2019-02-21 15:30:26
#   Description :
#
#================================================================

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOV3

class YoloTest(object):
    def __init__(self):
        self.input_size       = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold  = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path  = cfg.TEST.ANNOT_PATH
        self.weight_file      = cfg.TEST.WEIGHT_FILE
        self.write_image      = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label       = cfg.TEST.SHOW_LABEL

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')

        model = YOLOV3(self.input_data, self.trainable)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input_data: image_data,
                self.trainable: False
            }
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes

    def evaluate(self):
        predicted_dir_path = './mAP/predicted'
        ground_truth_dir_path = './mAP/ground-truth'
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(self.write_image_path)

        with open(self.annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)
                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

                if len(bbox_data_gt) == 0:
                    bboxes_gt=[]
                    classes_gt=[]
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

                print('=> ground truth of %s:' % image_name)
                num_bbox_gt = len(bboxes_gt)
                with open(ground_truth_path, 'w') as f:
                    for i in range(num_bbox_gt):
                        class_name = self.classes[classes_gt[i]]
                        xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                        bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())
                print('=> predict result of %s:' % image_name)
                predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
                bboxes_pr = self.predict(image)

                if self.write_image:
                    image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                    cv2.imwrite(self.write_image_path+image_name, image)

                with open(predict_result_path, 'w') as f:
                    for bbox in bboxes_pr:
                        coor = np.array(bbox[:4], dtype=np.int32)
                        score = bbox[4]
                        class_ind = int(bbox[5])
                        class_name = self.classes[class_ind]
                        score = '%.4f' % score
                        xmin, ymin, xmax, ymax = list(map(str, coor))
                        bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())

    def voc_2012_test(self, voc2012_test_path):

        img_inds_file = os.path.join(voc2012_test_path, 'ImageSets', 'Main', 'test.txt')
        with open(img_inds_file, 'r') as f:
            txt = f.readlines()
            image_inds = [line.strip() for line in txt]

        results_path = 'results/VOC2012/Main'
        if os.path.exists(results_path):
            shutil.rmtree(results_path)
        os.makedirs(results_path)

        for image_ind in image_inds:
            image_path = os.path.join(voc2012_test_path, 'JPEGImages', image_ind + '.jpg')
            image = cv2.imread(image_path)

            print('predict result of %s:' % image_ind)
            bboxes_pr = self.predict(image)
            for bbox in bboxes_pr:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = self.classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox_mess = ' '.join([image_ind, score, xmin, ymin, xmax, ymax]) + '\n'
                with open(os.path.join(results_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())


if __name__ == '__main__': YoloTest().evaluate()
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : evaluate.py
#   Author      : YunYang1994
#   Created date: 2019-02-21 15:30:26
#   Description :
#
#================================================================

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOV3

class YoloTest(object):
    def __init__(self):
        self.input_size       = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes      = len(self.classes)
        self.anchors          = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold  = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold    = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path  = cfg.TEST.ANNOT_PATH
        self.weight_file      = cfg.TEST.WEIGHT_FILE
        self.write_image      = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label       = cfg.TEST.SHOW_LABEL

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')

        model = YOLOV3(self.input_data, self.trainable)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        self.sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input_data: image_data,
                self.trainable: False
            }
        )

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes

    def evaluate(self):
        predicted_dir_path = './mAP/predicted'
        ground_truth_dir_path = './mAP/ground-truth'
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(self.write_image_path)

        with open(self.annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)
                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

                if len(bbox_data_gt) == 0:
                    bboxes_gt=[]
                    classes_gt=[]
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

                print('=> ground truth of %s:' % image_name)
                num_bbox_gt = len(bboxes_gt)
                with open(ground_truth_path, 'w') as f:
                    for i in range(num_bbox_gt):
                        class_name = self.classes[classes_gt[i]]
                        xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                        bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())
                print('=> predict result of %s:' % image_name)
                predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
                bboxes_pr = self.predict(image)

                if self.write_image:
                    image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                    cv2.imwrite(self.write_image_path+image_name, image)

                with open(predict_result_path, 'w') as f:
                    for bbox in bboxes_pr:
                        coor = np.array(bbox[:4], dtype=np.int32)
                        score = bbox[4]
                        class_ind = int(bbox[5])
                        class_name = self.classes[class_ind]
                        score = '%.4f' % score
                        xmin, ymin, xmax, ymax = list(map(str, coor))
                        bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())

    def voc_2012_test(self, voc2012_test_path):

        img_inds_file = os.path.join(voc2012_test_path, 'ImageSets', 'Main', 'test.txt')
        with open(img_inds_file, 'r') as f:
            txt = f.readlines()
            image_inds = [line.strip() for line in txt]

        results_path = 'results/VOC2012/Main'
        if os.path.exists(results_path):
            shutil.rmtree(results_path)
        os.makedirs(results_path)

        for image_ind in image_inds:
            image_path = os.path.join(voc2012_test_path, 'JPEGImages', image_ind + '.jpg')
            image = cv2.imread(image_path)

            print('predict result of %s:' % image_ind)
            bboxes_pr = self.predict(image)
            for bbox in bboxes_pr:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = self.classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox_mess = ' '.join([image_ind, score, xmin, ymin, xmax, ymax]) + '\n'
                with open(os.path.join(results_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())


if __name__ == '__main__': YoloTest().evaluate()
def calculat_parameter1(x, f0, fn):
    n = len(x)
    h = []
    for i in range(1, n):
        h.append(x[i][0] - x[i-1][0])
    u = [1]
    for i in range(1, n-1)[::-1]:
        u.insert(0, h[i-1] / (h[i-1] + h[i]))
    lamda = [1]
    for i in range(1, n-1):
        lamda.append(h[i] / (h[i-1] + h[i]))
    d = []
    for i in range(0, n):
        if i == 0:
            d.append(6 * (f([x[i], x[i+1]]) - f0) / h[0])
        elif i == n-1:
            d.append(6 * (fn - f([x[i-1], x[i]])) / h[i-1])
        else:
            d.append(6*f([x[i-1], x[i], x[i+1]]))

    return h, u, lamda, d
def calculat_parameter2(x, M0, Mn):
    n = len(x)
    h = []
    for i in range(1, n):
        h.append(x[i][0] - x[i-1][0])
    u = [0]
    for i in range(1, n-1)[::-1]:
        u.insert(0, h[i-1] / (h[i-1] + h[i]))
    lamda = [0]
    for i in range(1, n-1):
        lamda.append(h[i] / (h[i-1] + h[i]))
    d = []
    for i in range(0, n):
        if i == 0:
            d.append(2 * M0)
        elif i == n-1:
            d.append(2 * Mn)
        else:
            d.append(6*f([x[i-1], x[i], x[i+1]]))

    return h, u, lamda, d

def calculat_M(lamda, u, d):
    n = len(lamda)
    A = 2 * np.eye(n+1, dtype="float")

    for i in range(n):  # 构造矩阵A
        A[i][i+1] = lamda[i]
        A[i+1][i] = u[i]

    b = np.array([d]).transpose()
    M = np.dot(np.linalg.inv(A), b)
    return M

def s(x0, x1, M0, M1, x):
    h = x1[0] - x0[0]
    return M0 * pow((x1[0] - x), 3) / (6 * h) + M1 * pow((x - x0[0]), 3) / (6 * h) + (x0[1] - M0 * h * h / 6) * (
    x1[0] - x) / h + (x1[1] - M1 * h * h / 6) * (x - x0[0]) / h

def plot_s(x, M, color="#ff0000", label="方式"):
    n = len(x)

    plt.figure(1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('三次样条插值')
    for i in range(1, n):
        x0, x1 = x[i-1], x[i]
        plt.plot([x0[0], x1[0]], [x0[1], x1[1]], 'xc')
        M0, M1 = M[i-1], M[i]
        x_vector = np.linspace(x0[0], x1[0], 200)
        y_vector = s(x0, x1, M0, M1, x_vector).transpose()
        l, = plt.plot(x_vector, y_vector, color=color, lw=1)
    l.set_label(label) 


x = [(0, 1), (1, 0), (2, 0), (3, 1), (4, 2), (5, 2), (6, 1)]  # x点集

h1, u1, lamda1, d1 = calculat_parameter1(x, -0.6, -1.8)  # 第一类边界
M1 = calculat_M(lamda1, u1, d1)
plot_s(x, M1, color="#a012c5", label="第一类边界")
h2, u2, lamda2, d2 = calculat_parameter2(x, 1, -1)  # 第二类边界
M2 = calculat_M(lamda2, u2, d2)
plot_s(x, M2, color="#05a0ff", label="第二类边界")
h3, u3, lamda3, d3 = calculat_parameter2(x, 0, 0)  
