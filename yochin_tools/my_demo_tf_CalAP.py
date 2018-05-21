# coding=utf-8
# #!/usr/bin/env python

import os
import sys
import struct
import copy
sys.path.append('/home/yochin/Faster-RCNN_TF/lib')
# sys.path.append('/usr/lib/python2.7/dist-packages')

from networks.factory import get_network
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random
from xml.etree.ElementTree import Element, dump, SubElement, ElementTree
from datetime import datetime

import yo_network_info  # including info of net (class_name, num_classes)
# sys.path.append('/home/yochin/Faster-RCNN_TF/yochin_tools/Cal_mAP')
import Cal_mAP.check_mAP_V2 as check_mAP_V2

CONF_THRESH = 0.6
NMS_THRESH = 0.3

CLASSES = yo_network_info.CLASSES
NUM_CLASSES = yo_network_info.NUM_CLASSES

def list_files(path, ext):
    filelist = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            if name.endswith(ext):
                filelist.append(name)
    return filelist

def demo_all(sess, snet, im, strEstPathname):
    # scalefactor = 300. / float(min(im.shape[0], im.shape[1]))
    # tw = int(im.shape[1] * scalefactor)
    # th = int(im.shape[0] * scalefactor)
    # im = cv2.resize(im, (tw, th))

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('\t\t\t\tDetection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # im = im[:, :, (2, 1, 0)]  # for plt

    fontFace = cv2.FONT_HERSHEY_PLAIN;
    fontScale = 2;
    fontThickness = 2;

    if len(strEstPathname) > 0:
        tag_anno = Element('annotation')

    im_org = im.copy()

    # Visualize detections for each class
    for cls_ind, class_name in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

        if len(inds) > 0:
            for i in inds:
                bbox = dets[i, :4]      # [xmin, ymin, xmax, ymax]
                score = dets[i, -1]

                if score > 0.8:
                    fontColor = (255,0,0)
                    cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), fontColor, fontThickness)
                elif score > 0.6:
                    fontColor = (0, 255, 0)
                    cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), fontColor, fontThickness)
                else:
                    fontColor = (255, 255, 255)
                    cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), fontColor, fontThickness)

                cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (int(bbox[0]), int(bbox[1] - 2)), fontFace, fontScale, fontColor, thickness = fontThickness)


                print('{:s} {:.3f} {:d}'.format(class_name, score, cls_ind))

                if len(strEstPathname) > 0:
                    tag_object = Element('object')
                    SubElement(tag_object, 'name').text = class_name
                    SubElement(tag_object, 'score').text = str(score)
                    tag_bndbox = Element('bndbox')
                    SubElement(tag_bndbox, 'xmin').text = str(int(bbox[0]))
                    SubElement(tag_bndbox, 'ymin').text = str(int(bbox[1]))
                    SubElement(tag_bndbox, 'xmax').text = str(int(bbox[2]))
                    SubElement(tag_bndbox, 'ymax').text = str(int(bbox[3]))
                    tag_anno.append(tag_object)
                    tag_object.append(tag_bndbox)


    cv2.imshow('display', im)
    cv2.waitKey(10)

    if len(strEstPathname) > 0:
        cv2.imwrite(strEstPathname + '_est.jpg', im)
        ElementTree(tag_anno).write(strEstPathname)

    return im

def read_list_linebyline(fname):
    with open(fname) as fid:
        content = fid.readlines()
    content = [item.rstrip('\n') for item in content]

    return content


def write_list_linebyline(fname, thelist):
    fid = open(fname, 'w')

    for item in thelist:
        fid.write('%s\n' % (item))

    fid.close()

def yo_make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # for test
    cfg.NCLASSES = NUM_CLASSES
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.BBOX_REG = True   # Use BBox regressor
    # # Scales to use during testing (can list multiple scales)
    # # Each scale is the pixel size of an image's shortest side
    # cfg.TEST.SCALES = (600, 800, 1200, 1600, )
    # cfg.TEST.RPN_PRE_NMS_TOP_N = 12000
    # cfg.TEST.RPN_POST_NMS_TOP_N = 2000
    # # Max pixel size of the longest side of a scaled input image

    cfg.TEST.SCALES = (600,)  # only one for test

    tfArch = 'VGGnetslsv1_test'  # prototxt
    prjName = 'VGGnet-RealSingle_SynthMultiObj234'
    tfmodel = '../output/%s/train/VGGnet_fast_rcnn_iter_70000.ckpt'%prjName  # real db

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 1}))
    tf.device('')
    # load network
    net = get_network(tfArch)
    # load model
    print ('Loading network {:s}... '.format(tfArch))
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    print (' done.')

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(sess, net, im)

    if False:
        # TestDB - MultiObjectReal-181
        strBasePath = '/media/yochin/ModMan DB/ModMan DB/ETRI_HMI/real_data_label_set_Total/MultiObjectReal-181'
        strImageFolder  = os.path.join(strBasePath, 'Images/')       # 181 multiple objects image files
        strListfilePath = os.path.join(strBasePath, 'ImageSets/test.txt')
        strGndFolder    = os.path.join(strBasePath, 'Annotations/xml/')  # compare with this groundtruth

        strPathResult = '/home/yochin/Desktop/PlayGround_ModMan/EstResult/%s/MultiObjectReal-181/'%prjName         # will save the result xml
        strPathResultSummary = os.path.join(strPathResult, 'Summary')
    else:
        # TestDB - SingleObjectReal-1176
        strBasePath = '/media/yochin/0d71bed3-b968-40a1-a28d-bf12275c6299/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/data_realDB'
        strImageFolder  = os.path.join(strBasePath, 'Images')  # Single Object image files
        strListfilePath = os.path.join(strBasePath, 'ImageSets/readlSingleObject-10375/test.txt')
        strGndFolder    = os.path.join(strBasePath, 'Annotations/')  # compare with this groundtruth

        strPathResult = '../EstResult/%s/SingleObjectReal'%prjName         # will save the result xml
        strPathResultSummary = os.path.join(strPathResult, 'Summary')

    yo_make_directory(strPathResult)
    yo_make_directory(strPathResultSummary)

    if True:
        # From list file
        listTestFiles = read_list_linebyline(strListfilePath)
    else:
        # From image file
        listTestFiles1 = list_files(strImageFolder, 'bmp')
        listTestFiles2 = list_files(strImageFolder, 'jpg')
        listTestFiles = []
        listTestFiles.extend(listTestFiles1)
        listTestFiles.extend(listTestFiles2)

    for filenameExt in listTestFiles:
        filename = os.path.splitext(filenameExt)[0]
        filenameExt = filename + '.jpg'

        print 'Demo for data/demo/{}'.format(filenameExt)
        plt.close('all')
        if os.path.isfile(os.path.join(strImageFolder, filenameExt)):
            # Load the demo image
            im = cv2.imread(os.path.join(strImageFolder, filenameExt))
            demo_all(sess, net, im, os.path.join(strPathResult, filename + '_est.xml'))
            # cv2.waitKey(0)

    check_mAP_V2.check_mAP(CLASSES[1:]+('orange',), strListfilePath, strGndFolder, strPathResult, strPathResultSummary, False)
    # check_mAP_V2.check_mAP(CLASSES[1:]+('strawberry', 'papermate', 'highland', 'genuine', 'mark', 'expo', 'champion', 'apple', 'cup', 'banana', 'chiffon', 'crayola', 'scissors', 'tomatosoup', 'drill', 'waffle', 'ace', 'moncher', 'chococo', 'orange',), strListfilePath, strGndFolder, strPathResult, strPathResultSummary, False)
    check_mAP_V2.check_mAP(CLASSES[1:]+('orange',), strListfilePath, strGndFolder, strPathResult,
                           strPathResultSummary, False, check_only_recog = True)