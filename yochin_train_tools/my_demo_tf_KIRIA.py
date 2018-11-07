# coding=utf-8
# #!/usr/bin/env python

import os
import sys
import yochin_tools.yo_network_info
sys.path.append(os.path.join(yochin_tools.yo_network_info.PATH_BASE, 'lib'))

from networks.factory import get_network
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

#import caffe
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import argparse
import random
from shutil import copyfile

from xml.etree.ElementTree import Element, dump, SubElement, ElementTree
import time

CONF_THRESH = 0.6
NMS_THRESH = 0.3

# for DBv7
CLASSES = yochin_tools.yo_network_info.CLASSES
Candidate_CLASSES = yochin_tools.yo_network_info.Candidate_CLASSES

# Candidate_CLASSES = ('Ace', 'Apple', 'Champion', 'Cheezit', 'Chiffon',
#                     'Chococo', 'Crayola', 'Cup', 'Drill', 'Expo',
#                     'Genuine', 'Hammer', 'Highland', 'Mark', 'MasterChef',
#                     'Moncher', 'Mustard', 'Papermate', 'Peg', 'Scissors',
#                     'Sponge', 'TomatoSoup', 'Waffle', 'airplane', 'banana',
#                     'strawberry')

def list_files(path, ext):
    filelist = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            if name.endswith(ext):
                filelist.append(name)
    return filelist

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),       # (x, y)
                          bbox[2] - bbox[0],        # width
                          bbox[3] - bbox[1],        # height
                          fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

        print('{:s} {:.3f}'.format(class_name, score))

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(sess, net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

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
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # im = im[:, :, (2, 1, 0)]  # for plt

    fontFace = cv2.FONT_HERSHEY_PLAIN;
    fontScale = 2;
    fontThickness = 2;

    if len(strEstPathname) > 0:
        tag_anno = Element('annotation')

    cv2.namedWindow('display')
    plt.close('all')

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
                bbox = dets[i, :4]
                score = dets[i, -1]

                if class_name in Candidate_CLASSES:
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
    cv2.waitKey(1)

    # imgplot = plt.imshow(im)
    # plt.show()

    if len(strEstPathname) > 0:
        cv2.imwrite(strEstPathname + '_est.jpg', im)
        ElementTree(tag_anno).write(strEstPathname)

    return im

if __name__ == '__main__':
    USE_CAMERA = True

    # for trainig - just for ref
    # cfg.TRAIN.IMS_PER_BATCH: 2

    # for test
    cfg.NCLASSES = yochin_tools.yo_network_info.NUM_CLASSES
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.BBOX_REG = True   # Use BBox regressor
    cfg.TEST.SCALES = (600, )     # only one for test

    if USE_CAMERA == True:
        cap = cv2.VideoCapture(0)

    tfArch = 'VGGnetslsv1_test'  # prototxt
    # tfArch = 'Resnet50_test'  # prototxt
    # tfmodel = './output/Resnet50/train/VGGnet_fast_rcnn_iter_140000.ckpt'
    # tfmodel = './output/Resnet_scriptItself/voc_2007_trainval/Resnet50_iter_140000.ckpt'
    tfmodel = '../output/VGGnet_fast_rcnn_iter_70000.ckpt'


    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(tfArch)
    # load model
    print ('Loading network {:s}... '.format(tfArch))
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)
    print (' done.')

    # successive taken images
    num_dup_capture = 1



    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(sess, net, im)

    cnt = 0
    do_write_result = False
    if USE_CAMERA == True:
        while (True):
            ret, frame = cap.read()

            if ret is True:
                demo_all(sess, net, frame, './temp.xml')
            else:
                print('no frame\n')

            # cv2.imshow('frame', frame)

            input_key = cv2.waitKey(0)
            if input_key == ord('c'):
                print('capture')
                do_write_result = True
                copyfile('./temp.xml', '../ModMan_KIRIA/%04d_est.xml' % (cnt))
                copyfile('./temp.xml_est.jpg', '../ModMan_KIRIA/%04d_est.xml_est.jpg' % (cnt))
                cnt = cnt + 1
            elif input_key == ord('w'):
                CONF_THRESH = CONF_THRESH + 0.1
                print(CONF_THRESH)
            elif input_key == ord('s'):
                CONF_THRESH = CONF_THRESH - 0.1
                print(CONF_THRESH)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        strImageFolder = '/home/yochin/Desktop/ModMan_KIRIA/ObjectRecDet/2ndTrial/Total200/Images'
        strPathResult = '/home/yochin/Desktop/ModMan_KIRIA/ObjectRecDet/2ndTrial/Total200/estResult/'

        listTestFiles = list_files(strImageFolder, 'jpg')

        for filename in listTestFiles:
            im_name = filename

            print 'Demo for data/demo/{}'.format(im_name)
            if os.path.isfile(os.path.join(strImageFolder, im_name)):
                # Load the demo image
                im = cv2.imread(os.path.join(strImageFolder,im_name))
                demo_all(sess, net, im, os.path.join(strPathResult, os.path.splitext(filename)[0] + '_est.xml'))

                # cv2.waitKey(0)
