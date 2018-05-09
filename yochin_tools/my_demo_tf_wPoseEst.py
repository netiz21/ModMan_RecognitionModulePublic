# coding=utf-8
# #!/usr/bin/env python

import os
import sys
sys.path.append('/home/yochin/Faster-RCNN_TF/lib')
# sys.path.append('/usr/lib/python2.7/dist-packages')

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
from xml.etree.ElementTree import Element, dump, SubElement, ElementTree

CONF_THRESH = 0.6
NMS_THRESH = 0.3

# for DBv6
# CLASSES = ( '__background__',
#             'ace', #1
#             'champion',
#             'cheezit',
#             'chiffon',
#             'chococo', #5
#             'crayola',
#             'expo',
#             'genuine',
#             'highland',
#             'mark', #10
#             'moncher',
#             'papermate',
#             'waffle',
#             'cup',
#             'drill',
#             'mustard',
#             'scissors',
#             'tomatosoup') # 18

# # for DBv7
# CLASSES = ( '__background__',
#             'Ace', 'Apple', 'Champion', 'Cheezit', 'Chiffon',
#                     'Chococo', 'Crayola', 'Cup', 'Drill', 'Expo',
#                     'Genuine', 'Hammer', 'Highland', 'Mark', 'MasterChef',
#                     'Moncher', 'Mustard', 'Papermate', 'Peg', 'Scissors',
#                     'Sponge', 'TomatoSoup', 'Waffle', 'airplane', 'banana',
#                     'strawberry')
#
# Candidate_CLASSES = ('Ace', 'Apple', 'Cheezit', 'Chiffon', 'Crayola',
#                      'Drill', 'Genuine', 'Mustard', 'TomatoSoup', 'airplane')
#
# # for DBV11_10obj
# CLASSES_10obj = ['__background__',
#                  'Ace', 'Apple', 'Cheezit', 'Chiffon', 'Crayola',
#                  'Drill', 'Genuine', 'Mustard', 'TomatoSoup', 'airplane']
# CLASSES = CLASSES_10obj

# for realV1
CLASSES = ('__background__',
           'strawberry', 'Papermate', 'Highland', 'Genuine', 'Mark',
           'Expo', 'Champion', 'Orange', 'Apple', 'Cup',
           'banana', 'Chiffon', 'Crayola', 'Scissors', 'TomatoSoup',
           'Drill', 'Mustard', 'Waffle', 'Ace', 'airplane',
           'Moncher', 'Cheezit', 'Chococo'
)

Candidate_CLASSES = CLASSES

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')

NUM_CLASSES = len(CLASSES) # +1 for background

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
    fontScale = 1;
    thickness = 2;

    if len(strEstPathname) > 0:
        tag_anno = Element('annotation')

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
                        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0, 0), 2)
                    elif score > 0.6:
                        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0, 0), 2)
                    elif score > 0.4:
                        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0, 0), 2)

                    cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (int(bbox[0]), int(bbox[1] - 2)), fontFace, fontScale, (255,255,255))

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

    if len(strEstPathname) > 0:
        cv2.imwrite(strEstPathname + '_est.jpg', im)
        ElementTree(tag_anno).write(strEstPathname)

    return im

if __name__ == '__main__':
    USE_CAMERA = True

    # for trainig - just for ref
    # cfg.TRAIN.IMS_PER_BATCH: 2

    # for test
    cfg.NCLASSES = NUM_CLASSES
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.BBOX_REG = True   # Use BBox regressor
    cfg.TEST.SCALES = (600, )     # only one for test

    if USE_CAMERA == True:
        cap = cv2.VideoCapture(0)

    tfArch = 'VGGnetslsv1_test'  # prototxt
    # tfArch = 'Resnet50_test'  # prototxt
    # tfmodel = './output/Resnet50/train/VGGnet_fast_rcnn_iter_140000.ckpt'
    # tfmodel = './output/Resnet_scriptItself/voc_2007_trainval/Resnet50_iter_140000.ckpt'
    # tfmodel = '../output/VGGnet_140000_noFlipped_DBV10_train/train/VGGnet_fast_rcnn_iter_140000.ckpt'
    # tfmodel = '../output/VGGnet_70000_noFlipped_DB_RealV1_train/train/VGGnet_fast_rcnn_iter_70000.ckpt'
    # tfmodel = '../output/VGGnet_140000_noFlipped_DBV11_10obj_train/train/VGGnet_fast_rcnn_iter_140000.ckpt'


    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
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

    if USE_CAMERA == True:
        while (True):
            ret, frame = cap.read()

            if ret is True:
                demo_all(sess, net, frame, '')
            else:
                print('no frame\n')

            # cv2.imshow('frame', frame)

            input_key = cv2.waitKey(1)

            if input_key == ord('q'):
                break
            elif input_key == ord('w'):
                CONF_THRESH = CONF_THRESH + 0.1
                print(CONF_THRESH)
            elif input_key == ord('s'):
                CONF_THRESH = CONF_THRESH - 0.1
                print(CONF_THRESH)

        cap.release()
        cv2.destroyAllWindows()
    else:
        # strImageFolder = '/media/yochin/ModMan DB/ModMan DB/ETRI_HMI/ETRI_RGBD_CUBE DB/image/Try8/'
        # strImageFolder = '/media/yochin/ModMan DB/ModMan DB/ETRI_HMI/ETRI_RGBD_CUBE DB/image/Try6/'
        # strImageFolder = '/home/yochin/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/data/Images/'
        # strImageFolder = '/media/yochin/ModMan DB/ModMan DB/ETRI_HMI/Synthetic Test SceneV2(3rd yr)/TestSet/Total200/Images/'
        strImageFolder = '/home/yochin/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/data/Images'

        strPathResult = '/media/yochin/ModMan DB/ModMan DB/ETRI_HMI/Synthetic Test SceneV2(3rd yr)/TestSet/Total200/estResult/'

        listTestFiles = list_files(strImageFolder, 'jpg')

        for filename in listTestFiles:
            im_name = filename
            est_filename = os.path.splitext(filename)[0] + '_est.xml'

            print 'Demo for data/demo/{}'.format(im_name)
            plt.close('all')
            if os.path.isfile(os.path.join(strImageFolder, im_name)):
                # Load the demo image
                im = cv2.imread(os.path.join(strImageFolder, im_name))
                # demo_all(sess, net, im, strPathResult + est_filename)
                demo_all(sess, net, im, '')
                cv2.waitKey(0)
