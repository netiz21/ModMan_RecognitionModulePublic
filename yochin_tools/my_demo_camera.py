# coding=utf-8
# #!/usr/bin/env python

# default library
import os
import sys
import struct
import copy

import yo_network_info
import time
sys.path.append(os.path.join(yo_network_info.PATH_BASE, 'lib'))
# sys.path.append('/usr/lib/python2.7/dist-packages')

# Faster-RCNN_TF
from networks.factory import get_network
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

# ModMan module & Pose estimation
from PoseEst.Function_Pose_v3 import *
import math

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from xml.etree.ElementTree import Element, dump, SubElement, ElementTree
from datetime import datetime

import threading

# socket
from socket import *
import errno

# debugging
import logging
logging.basicConfig(level = logging.INFO)

# realsense
# import pyrealsense as pyrs

CLASSES = yo_network_info.CLASSES
Candidate_CLASSES = yo_network_info.Candidate_CLASSES
NUM_CLASSES = yo_network_info.NUM_CLASSES

CONF_THRESH = yo_network_info.DETECTION_TH
NMS_THRESH = 0.3

DO_WRITE_RESULT_IMG = False  # IMG name is the current timeshot.
DO_SAVE_DEBUGIMAGE = False

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def list_files(path, ext):
    filelist = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            if name.endswith(ext):
                filelist.append(name)
    return filelist

class ThreadedVideoCapture(object):
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.image = None
        self.ret = None
        self.STOP_CAPTURE = False

    def do_capture(self):
        while True:
            if self.STOP_CAPTURE is True:
                cv2.waitKey(10)
            else:
                # self.ret, self.image = cap.read()
                self.ret = cap.grab()
                cv2.waitKey(1)

    def get_image(self):
        self.STOP_CAPTURE = True

        self.ret, ret_image = cap.retrieve(self.ret)
        # ret_image = copy.copy(self.image)

        self.STOP_CAPTURE = False

        return self.ret, ret_image



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

        # print('{:s} {:.3f}'.format(class_name, score))

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo_all(sess, snet, im, strEstPathname, extMat=None, FeatureDB=None, CoorDB=None, GeoDB=None):
    ret_list_forKIST = []
    ret_list_BB = []

    # Detect all object classes and regress object bounds
    t = time.time()
    scores, boxes = im_detect(sess, net, im)
    # cv2.waitKey(1000)
    scores = np.zeros((2000, 23), dtype=np.float)
    boxes = np.zeros((2000, 92), dtype=np.float)
    elapsed = time.time() - t
    print('Detect: %.3f\n' % elapsed)
    # print ('\t\t\t\tDetection took {:.3f}s for '
    #        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # im = im[:, :, (2, 1, 0)]  # for plt

    t = time.time()
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

        # # make the result,
        # dets[0, :4] = [0, 0, im.shape[1], im.shape[0]]
        # dets[0, -1] = .99
        # class_name = 'ace'
        # inds = [0]

        if len(inds) > 0:
            for i in inds:
                bbox = dets[i, :4]      # [xmin, ymin, xmax, ymax]
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

                    cv2.putText(im, '{:s} {:.1f}'.format(class_name, score*100), (int(bbox[0]), int(bbox[1] - 2)), fontFace, fontScale, fontColor, thickness = fontThickness)

                    ret_list_BB.append({'bbox': bbox, 'score': score, 'name': class_name})


                    # print('{:s} {:.3f} {:d}'.format(class_name, score, cls_ind))

                    if extMat is not None:
                        FeatureDB2 = FeatureDB[Candidate_CLASSES.index(class_name)]
                        CoorDB2 = CoorDB[Candidate_CLASSES.index(class_name)]
                        GeoDB2 = GeoDB[Candidate_CLASSES.index(class_name)]

                        width = np.min([int(bbox[2]) - int(bbox[0]),10*int(bbox[0])])
                        height = np.min([int(bbox[3]) - int(bbox[1]),10*int(bbox[1])])

                        cropbox_ly = int(bbox[1]-height*0.1)
                        cropbox_ry = int(bbox[3]+height*0.1)
                        cropbox_lx = int(bbox[0]-width*0.1)
                        cropbox_rx = int(bbox[2]+width*0.1)

                        cropimg = im_org[cropbox_ly:cropbox_ry, cropbox_lx:cropbox_rx, :]

                        # timestamp = '%d_%d_%d_%d_%d_%d_%d' % (
                        # datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour,
                        # datetime.now().minute, datetime.now().second, datetime.now().microsecond)
                        # cv2.imwrite('./%s.jpg' % timestamp, cropimg)
                        # print('captured %s image', timestamp)

                        # cv2.namedWindow('crop')
                        # cv2.imshow('crop', cropimg)
                        # cv2.waitKey(0)
                        #
                        print('bbox:')
                        print(bbox)

                        init_coord = np.array([cropbox_lx,  cropbox_ly, 0])    # init_coord[x, y, -], lefttop_point
                        rmat, tvec = PoseEstimate(cropimg, FeatureDB2, CoorDB2, extMat, init_coord)

                        # fid = open('./%s.txt'%timestamp, 'w')
                        # print >> fid, init_coord, '\n\n', tvec, '\n\n', imgpts, '\n\n', objpts
                        # fid.close()

                        print(rmat)
                        print(tvec)
                        if rmat.sum() == 0 or np.isnan(rmat).any() or np.isnan(tvec).any() == True:
                            print('cannot find the pose information and fill with dummy values with all zeros')
                            # return for KIST
                            obj_info = {'object': class_name,
                                        'score': score,
                                        'RMat': rmat,
                                        'TVec': tvec * 0.0,
                                        'x_center': (bbox[0] + bbox[2]) / 2,
                                        'y_center': (bbox[1] + bbox[3]) / 2,
                                        'left': bbox[0],
                                        'top': bbox[1],
                                        'right': bbox[2],
                                        'bottom': bbox[3]
                                        }
                            ret_list_forKIST.append(obj_info)

                        else:
                            init_coord = np.array([0, 0, 0])
                            Result = cornerpointsTransform2(GeoDB2, rmat, tvec, extMat, init_coord)

                            # outDots = np.zeros((4,3))
                            # outtvecs = np.zeros((1,3))

                            # CCI = Candidate_CLASSES.index(class_name.lower())
                            # if current[CCI] < avgwindow:
                            #     current[CCI] = current[CCI] + 1
                            #     DotLog[:, :, current[CCI] - 1:current[CCI], CCI] = Result[:,:,None]
                            #     tvecLog[:, :, current[CCI] - 1:current[CCI], CCI] = tvec[:,:,None]
                            #     outDots = np.mean(DotLog[:, :, current[CCI] - 1:current[CCI], CCI], axis=2)
                            #     outtvecs = np.mean(tvecLog[:, :, current[CCI] - 1:current[CCI], CCI], axis=2)
                            # else:
                            #     DotLog[:, :, 0:current[CCI]-1, CCI] = DotLog[:, :, 1:current[CCI] + 1, CCI]
                            #     tvecLog[:, :, 0:current[CCI]-1, CCI] = tvecLog[:, :, 1:current[CCI] + 1, CCI]
                            #     DotLog[:, :, current[CCI] - 1:current[CCI],CCI] = Result[:, :, None]
                            #     tvecLog[:, :, current[CCI] - 1:current[CCI],CCI] = tvec[:, :, None]
                            #     outDots = np.mean(DotLog[:, :, :, CCI], axis=2)
                            #     outtvecs = np.mean(tvecLog[:, :, :, CCI], axis=2)
                            # Result = outDots
                            # tvec = outtvecs

                            # return for KIST
                            obj_info = {'object': class_name,
                                        'score': score,
                                        'RMat': rmat,
                                        'TVec': tvec,
                                        'x_center': (bbox[0] + bbox[2]) / 2,
                                        'y_center': (bbox[1] + bbox[3]) / 2,
                                        'left': bbox[0],
                                        'top': bbox[1],
                                        'right': bbox[2],
                                        'bottom': bbox[3]
                                        }
                            ret_list_forKIST.append(obj_info)

                            print('\tRot info: ')
                            print(rmat)
                            # print('\tTrn info:\n\t\tx: %d\n\t\ty: %d\n\t\tz: %d' % (tvec[1] * 0.1, -tvec[0] * 0.1, tvec[2] * 0.1))      # *0.1 --> mm to cm
                            print('\tTvec info: ')
                            print(tvec)
                            #print('\tTrn info:\n\t\tx: %d\n\t\ty: %d\n\t\tz: %d' % (tvec[1]/tvec[0], -tvec[0]//tvec[0], tvec[2]/tvec[0]))

                            # draw axis
                            drawLineIndeces = ((0, 1), (0, 2), (0, 3))
                            colorList = ((255, 0, 0), (0, 255, 0), (0, 0, 255))  # z, y, x
                            for (idxStart, idxEnd), color in zip(drawLineIndeces, colorList):
                                cv2.line(im, (int(Result[idxStart][0]), int(Result[idxStart][1])),
                                         (int(Result[idxEnd][0]), int(Result[idxEnd][1])), color, thickness=4)

                            # draw point
                            for ptDisp in Result:
                                cv2.circle(im, (int(ptDisp[0]), int(ptDisp[1])), 5, (255,255,255,0), -1)

                            # draw center position from a camera (mm -> cm by x 0.1)
                            cv2.putText(im, '(%d, %d, %d)'%(tvec[1] * 0.1, -tvec[0] * 0.1, tvec[2] * 0.1), (int(Result[0][0]), int(Result[0][1])), fontFace, fontScale, fontColor, thickness=fontThickness)


                            # draw gripping point based the database
                            ptGPs = np.array(yo_network_info.ListGrippingPoint[class_name.lower()])
                            if len(ptGPs) > 0:
                                pt2DGPs = computeTransfrom(ptGPs, rmat, tvec, extMat, init_coord)
                                pt2DGPs = np.transpose(pt2DGPs)
                                for ptDisp in pt2DGPs:
                                    cv2.circle(im, (int(ptDisp[0]), int(ptDisp[1])), 3, (250,150,100,0), -1)

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

                        if extMat is not None and rmat.sum() != 0 and np.isnan(rmat).any() != True:
                            SubElement(tag_object, 'rotation_matrix').text = str(rmat);
                            SubElement(tag_object, 'traslation_vector').text = str(tvec);

                            xyz = rotationMatrixToEulerAngles(rmat)
                            SubElement(tag_object, 'EulerAngles').text = str(xyz)
    cv2.namedWindow('display', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('display', im)
    cv2.waitKey(10)

    elapsed = time.time() - t
    print('Drawing: %.3f\n' % elapsed)

    if len(strEstPathname) > 0:
        cv2.imwrite(strEstPathname + '_est.jpg', im)
        ElementTree(tag_anno).write(strEstPathname)

    return im, ret_list_forKIST, ret_list_BB

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

def get_curtime_as_string():
    timestamp = '%d_%d_%d_%d_%d_%d_%d' % (
        datetime.now().year,
        datetime.now().month,
        datetime.now().day,
        datetime.now().hour,
        datetime.now().minute,
        datetime.now().second,
        datetime.now().microsecond
    )

    return timestamp

gmp_ix, gmp_iy = -1, -1
gmp_event = -1
def get_mouse_position(event, x, y, flags, param):
    global gmp_ix, gmp_iy, gmp_event
    if event == cv2.EVENT_LBUTTONDOWN:
        gmp_ix, gmp_iy, gmp_event = x, y, event

def getCamIntParams(nameDevice):
    extMat = np.zeros((3, 3))
    if nameDevice == 'SKKU':
        print('Set SKKU Camera intrinsic param')
        # # value from SKKU
        # extMat[0, 0] = 2398.9479815687  # fy
        # extMat[1, 1] = 2399.8176947948  # fx
        # extMat[0, 2] = 492.5551416881  # py
        # extMat[1, 2] = 622.6012000501  # px
        # extMat[2, 2] = 1

        # # SKKU Camera - Calculate at KIST by myself - 1.5m distance btw camera and obj
        # extMat[0, 0] = 2532.719  # fy
        # extMat[1, 1] = 2534.086  # fx
        # extMat[0, 2] = 493.783  # py
        # extMat[1, 2] = 627.673  # px
        # extMat[2, 2] = 1

        # SKKU Camera - Calculate at KIST by myself - 0.7m distance btw camera and obj
        extMat[0, 0] = 2559.335  # fy
        extMat[1, 1] = 2559.955  # fx
        extMat[0, 2] = 467.309  # py
        extMat[1, 2] = 605.107  # px
        extMat[2, 2] = 1

    elif nameDevice == 'C920':
        # Logitech C920
        print('Set MS C920 Webcam intrinsic param')
        extMat[0, 0] = 603.257  # fy
        extMat[1, 1] = 603.475  # fx
        extMat[0, 2] = 237.936  # py
        extMat[1, 2] = 323.122  # px
        extMat[2, 2] = 1
    elif nameDevice == 'KinectV2':
        # KinectV2
        print('Set KINECTv2 intrinsic param')
        extMat[0, 0] = 1101.658925  # fy
        extMat[1, 1] = 1102.363184  # fx
        extMat[0, 2] = 513.583814  # py
        extMat[1, 2] = 937.202226  # px
        extMat[2, 2] = 1
    elif nameDevice == 'iphone6S':
        print('Set iPhone6S intrinsic param')
        extMat[0, 0] = 1122.555  # fy
        extMat[1, 1] = 1124.040  # fx
        extMat[0, 2] = 371.990  # py
        extMat[1, 2] = 661.668  # px
        extMat[2, 2] = 1
    elif nameDevice == 'iPadPro10.5':
        print('Set iPad intrinsic param')
        extMat[0, 0] = 2398.881  # fy
        extMat[1, 1] = 2400.970  # fx
        extMat[0, 2] = 815.355  # py
        extMat[1, 2] = 1096.002  # px
        extMat[2, 2] = 1
    elif nameDevice == 'SR300':
        print('Set SR300 intrinsic param')
        extMat[0, 0] = 620.330732  # fy
        extMat[1, 1] = 617.397217  # fx
        extMat[0, 2] = 241.907642  # py
        extMat[1, 2] = 321.188038  # px
        extMat[2, 2] = 1
    elif nameDevice == 'client':
        print('Get intrinsic param from the client PC')

    return extMat





if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    '''
    Settings
    '''
    INPUT_TYPE = 0      # 0: WebCamera,
                        # 1: Network input from SKKU CAM,
                        # 2: Image,
                        # 3: Video,
                        # 4: Network input from ETRI(as client),
                        # 5: working as server for IPad,
                        # 6: Realsense Camera
                        # 7: working as server for SR300
    USE_DETECTION = True
    USE_POSEESTIMATE = True


    AR_IP = '129.254.87.77'
    AR_PORT = 8020

    if USE_POSEESTIMATE == True:    # Cam Intrinsic Params Settings
        extMat = getCamIntParams('C920')
        # extMat = getCamIntParams('client')


    DO_WRITE_RESULT_AVI = False
    name_output_avi = 'output.avi'

    # these options are on the top page.
    # DO_WRITE_RESULT_IMG = False     # IMG name is the current timeshot.
    # DO_SAVE_DEBUGIMAGE = False

    # for detection
    cfg.NCLASSES = NUM_CLASSES
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.BBOX_REG = True  # Use BBox regressor
    # # Scales to use during testing (can list multiple scales)
    # # Each scale is the pixel size of an image's shortest side
    cfg.TEST.SCALES = (688,)  # only one for test  ,480, 576, 688, 864, 1200
    cfg.TEST.MAX_SIZE = 1000
    cfg.TEST.RPN_NMS_THRESH = 0.7

    ## Number of top scoring boxes to keep before apply NMS to RPN proposals
    # cfg.TEST.RPN_PRE_NMS_TOP_N = 6000
    cfg.TEST.RPN_PRE_NMS_TOP_N = 12000
    ## Number of top scoring boxes to keep after applying NMS to RPN proposals
    # cfg.TEST.RPN_POST_NMS_TOP_N = 300
    cfg.TEST.RPN_POST_NMS_TOP_N = 2000
    CONF_THRESH = 0.8
    # cfg.TEST.MAX_SIZE = 2000
    print(cfg.TEST)

    if INPUT_TYPE == 2:  # Image
        avgwindow = 1
    else:
        avgwindow = 0   # parameter to show pose estimation in stable


    '''
    Write Result
    '''
    if DO_WRITE_RESULT_AVI == True:
        # Define the codec and create VideoWriter object
        frame_width = 720
        frame_height = 404
        outavi = cv2.VideoWriter(name_output_avi, cv2.VideoWriter_fourcc('M','J','P','G'), 15.0, (frame_width, frame_height))


    # this is for pose estimation
    objNum = len(Candidate_CLASSES)
    DotLog = np.zeros((4, 3, avgwindow, objNum))
    current = np.zeros((objNum), dtype="int")
    tvecLog = np.zeros((3, 1, avgwindow, objNum))

    FeatureDB = []
    CoorDB = []
    keypointDB = []
    GeoDB = []
    if USE_POSEESTIMATE == True:
        gap = 4

        for ith, obj in enumerate(Candidate_CLASSES):
            temp_FeatureDB, temp_CoorDB, temp_keypointDB = ReadDB(obj)
            FeatureDB.append(temp_FeatureDB[::gap,:])
            CoorDB.append(temp_CoorDB[::gap,:])
            # keypointDB.extend(temp_keypointDB)

            # read plotting information of the object
            # DBv1
            filelist2 = os.listdir(os.path.join(basePath, obj.lower(), 'geom2'))
            filelist2 = np.sort(filelist2)
            strTRSet = os.path.join(os.path.join(basePath, obj.lower(), 'geom2', filelist2[0]))
            # DBv2
            # strTRSet = os.path.join(basePath, '%s-rotate-geom_00_00_50_400mat.mat'%obj)
            ftr = h5py.File(strTRSet, 'r')
            GeoDB.append(np.transpose(np.array(ftr['img']), [2, 1, 0]))

    if INPUT_TYPE == 0:
        for i_temp in range(1, 10):
            cap = cv2.VideoCapture(i_temp)
            if cap.isOpened() == True:
                print('camera %d is openned'%i_temp)
                break
        # cap.set(3, 640*2)
        # cap.set(4, 480*2)

        threadCamera = ThreadedVideoCapture(cam_id=i_temp)

        threading.Thread(target=threadCamera.do_capture, \
                         args=() \
                         ).start()


    elif INPUT_TYPE == 6:
        serv = pyrs.Service()
        pyrs.start()

        '''
        pyrc.stream
        .color
        .depth
        .cad (color aligned on depth)
        .dac (depth aligned on color)
        '''

        cam = serv.Device(device_id=0, streams=[pyrs.stream.ColorStream(fps=60),
                                                pyrs.stream.DepthStream(fps=60)
                                                # pyrs.stream.CADStream(fps=60),
                                                # pyrs.stream.DACStream(fps=60)
                                                ])
        scale = cam.depth_scale * 1000
    elif INPUT_TYPE == 3:
        cap = cv2.VideoCapture('/home/yochin/Desktop/demo_avi.mp4')

    tfArch = 'VGGnetslsv1_test'  # prototxt
    tfmodel = '../models/VGGnet_fast_rcnn_iter_70000.ckpt'
    # tfmodel = '../output/VGGnet-RealSingle_SynthMultiObj234/train/VGGnet_fast_rcnn_iter_70000.ckpt'

    # init session
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.7)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 1}))
    # , device_count = {'GPU': 0}
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

    if INPUT_TYPE == 6:
        cv2.namedWindow('color')
        cv2.namedWindow('depth')
        # cv2.namedWindow('CAD')
        # cv2.namedWindow('DAC')

        cv2.setMouseCallback('depth', get_mouse_position)

        while (True):
            cam.wait_for_frames()
            # print(cam.color)
            current_color = cam.color[:, :, ::-1]
            current_depth = cam.depth * scale
            # current_cad = cam.cad[:, :, ::-1]
            # current_dac = cam.dac * scale

            cv2.imshow('color', current_color)
            cv2.imshow('depth', current_depth / 1000)
            # cv2.imshow('CAD', current_cad)
            # cv2.imshow('DAC', current_dac / 1000)

            cv2.waitKey(10)

            # print(current_depth[current_depth.shape[0]/2, current_depth.shape[1]/2])

            if gmp_event == cv2.EVENT_LBUTTONDOWN:
                gmp_event = -1
                # print(gmp_iy, gmp_ix)
                print(current_depth[gmp_iy, gmp_ix])

            if USE_POSEESTIMATE is True:
                im, list_objs_forKIST, ret_list_BB = demo_all(sess, net, np.array(current_color), '', extMat, FeatureDB, CoorDB, GeoDB)
            else:
                im, _, _ = demo_all(sess, net, np.array(current_color), '')



            # print(list_BB)

        cam.stop()
        serv.stop()

    elif INPUT_TYPE == 0 or INPUT_TYPE == 3:
        while (True):
            t = time.time()
            # ret, frame = cap.read()
            # for i in range(4):
            #     ret = cap.grab()
            # ret, frame = cap.retrieve(ret)
            # ret = True
            ret, frame = threadCamera.get_image()

            elapsed = time.time() - t
            print('Capture: %.3f\n'%elapsed)

            # frame = np.rot90(frame, k=3)
            #
            # cv2.namedWindow('Input')
            # cv2.imshow('Input', frame)
            # cv2.waitKey

            # frame = cv2.flip(frame, 1)

            if ret is True:
                if DO_SAVE_DEBUGIMAGE == True:
                    cv2.imwrite('./debug_img.png', frame)

                if USE_POSEESTIMATE is True:
                    im, list_objs_forKIST, _ = demo_all(sess, net, frame, '', extMat, FeatureDB, CoorDB, GeoDB)
                else:
                    im, _, _ = demo_all(sess, net, frame, '')

                if DO_WRITE_RESULT_AVI == True:
                    print(im.shape)
                    outavi.write(im)

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
                elif input_key == ord('c'):
                    timestamp = '%d_%d_%d_%d_%d_%d_%d'%(datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour, datetime.now().minute, datetime.now().second, datetime.now().microsecond)
                    cv2.imwrite('./%s.png'%timestamp, frame)
                    print('captured %s image', timestamp)
                elif input_key == ord('p'):
                    USE_POSEESTIMATE = 1 - USE_POSEESTIMATE
                    print('USE_POSEESTIMATE: %d'%(USE_POSEESTIMATE))
            else:
                print('no frame\n')

            # cap.grab()


        if DO_WRITE_RESULT_AVI == True:
            outavi.release()
            print('outavi is released')
        cap.release()
        cv2.destroyAllWindows()
    elif INPUT_TYPE == 1:
        # # as a client, connect to a KIST server
        KIST_IP = '192.168.137.13'
        KIST_PORT= 8000
        KIST_BUFSIZE = 4096
        KIST_ADDR = (KIST_IP, KIST_PORT)
        KIST_clientSocket = socket(AF_INET, SOCK_STREAM)

        try:
            print('trying to connect the KIST server')
            KIST_clientSocket.connect(KIST_ADDR)
        except Exception as e:
            print('cannot connect the KIST server')
            sys.exit()
        print('connected to the KIST server')

        # if we receive the call from kist,
        # send the request to SKKU server for receiving an image
        # then calculate the rmat, tvec
        # tehn send the result to kist
        SKKU_IMG_WIDTH = 1280
        SKKU_IMG_HEIGHT = 960
        SKKU_IP = '192.168.137.5'
        SKKU_PORT = 9000
        SKKU_NET_BUFSIZE = (SKKU_IMG_WIDTH * SKKU_IMG_HEIGHT * 3 + 6) * 10
        SKKU_ADDR = (SKKU_IP, SKKU_PORT)
        clientSocket = socket(AF_INET, SOCK_STREAM)

        try:
            print('trying to connect the SKKU server')
            clientSocket.connect(SKKU_ADDR)
        except Exception as e:
            print('cannot connect the SKKU server')
            sys.exit()
        print('connected to the SKKU server')

        while True:
            data = KIST_clientSocket.recv(KIST_BUFSIZE)
            n_stacked_result = 0
            # data = 'ETRI'

            if 'ETRI' in data:
                print('receive _ETRI_ from KIST')

                while True:
                    ing_rcv = False
                    len_rcv = 0

                    print('start SKKU')
                    clientSocket.send('3')  # 1 for streamming, 2 for stopping, 3 for one-shot image
                    print('send _3_ to SKKU server')
                    do_capture_success = False

                    while True:
                        data = clientSocket.recv(SKKU_NET_BUFSIZE)

                        if ing_rcv == False and 'MMS' in data:
                            ing_rcv = True

                            index_start = data.index('MMS')

                            fid_bin = open('./skku_img.bin', 'wb')
                            fid_bin.write(data[index_start+3:])
                            len_rcv = len_rcv + len(data[index_start+3:])
                        elif ing_rcv == True:
                            if ('MME' in data) and (data.index('MME') + len_rcv == 960*1280*3):
                                ing_rcv = False
                                index_end = data.index('MME')
                                fid_bin.write(data[:index_end])
                                fid_bin.close()

                                len_rcv = len_rcv + len(data[:index_end])
                                # print(len_rcv)
                                len_rcv = 0

                                img = np.fromfile('./skku_img.bin', dtype='uint8')

                                if len(img) != 960*1280*3:
                                    print('captured image size is not 960 x 1280 x 3')
                                    do_capture_success = False
                                else:
                                    img = img.reshape(960, 1280, 3)
                                    do_capture_success = True

                                break
                            else:
                                fid_bin.write(data)
                                len_rcv = len_rcv + len(data)

                        # print(data)

                    print('end SKKU')

                    # # do ETRI job
                    # # # temp
                    # img = np.fromfile('./skku_img.bin', dtype='uint8')
                    # img = img.reshape(960, 1280, 3)
                    # do_capture_success = True

                    if do_capture_success is True:
                        # img = cv2.flip(img, 0)
                        # img2 = img.copy()
                        # img = (((img.astype(float) / 255.) ** 1.6) * 255).astype('uint8')  # correction
                        # img[:,:,0] = (img[:,:,0].astype(float)/1.1).astype('uint8')

                        # timestamp = '%d_%d_%d_%d_%d_%d_%d' % (
                        #     datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour,
                        #     datetime.now().minute,
                        #     datetime.now().second, datetime.now().microsecond)
                        # cv2.imwrite('./%s.jpg' % timestamp, img)
                        # print('captured %s image', timestamp)

                        if USE_POSEESTIMATE is True:
                            img, list_objs_forKIST = demo_all(sess, net, img, '', extMat, FeatureDB, CoorDB, GeoDB)
                        else:
                            demo_all(sess, net, img, '')
                    else:
                        print('no frame\n')

                    do_capture_success = False

                    # cv2.imshow('frame', frame)

                    input_key = cv2.waitKey(10)

                    if len(list_objs_forKIST) > 0:
                        n_stacked_result = n_stacked_result + 1

                        if n_stacked_result >= avgwindow:
                            break

                if input_key == ord('q'):
                    clientSocket.send('2')  # 1 for streamming, 2 for stopping, 3 for one-shot image
                    print('send 2\n')

                    break

                # send to kist
                # 4 byte: num of object (int)
                # 1 byte: obj ID (char)
                # 4 * (9 + 3) = 48 bytes = rot + trans mat
                # 4 * 2 = 8 bytes = x, y

                # # temp - start
                # list_objs_forKIST = []
                # obj_info = {'object': 'Hello', 'score': 100., 'RMat': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 'TVec': np.array([[10], [20], [30]]),
                #             'x_center': 100, 'y_center': 200}
                # list_objs_forKIST.append(obj_info)
                # # temp - end

                msg = 'MMS'
                msg = msg + struct.pack('i', len(list_objs_forKIST))  # int

                for obj_KIST in list_objs_forKIST:
                    msg = msg + struct.pack('c', obj_KIST['object'][0])

                    for j_RMat in range(0, 3):
                        for i_RMat in range(0, 3):
                            msg = msg + struct.pack('f', obj_KIST['RMat'][i_RMat][j_RMat])

                    for j_TVec in range(0, 3):
                        msg = msg + struct.pack('f', obj_KIST['TVec'][j_TVec][0])

                    msg = msg + struct.pack('i', int(obj_KIST['x_center'])) + struct.pack('i', int(obj_KIST['y_center']))

                msg = msg + 'MME'


                # for list_objs
                KIST_clientSocket.send(msg)
                print('send _%s_ to KIST'%(msg))
    elif INPUT_TYPE == 2:
        # strImageFolder = '/media/yochin/ModMan_DB/ModMan_DB/ETRI_HMI/real_data_label_set_Total/ModMan_SLSv1/data_realDB/Images'  # Single Object image files
        # strImageFolder = '../debug_dropbox_upload/test'
        strImageFolder = '../realDB'
        strPathResult = './dummy'

        if not os.path.exists(strPathResult):
            os.makedirs(strPathResult)

        if False:
            # From list file
            strListFilePath = '/media/yochin/0d71bed3-b968-40a1-a28d-bf12275c6299/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/data_realDB/ImageSets/readlSingleObject-10375/test.txt'
            listTestFiles = read_list_linebyline(strListFilePath)
        else:
            # From image file
            listTestFiles1 = list_files(strImageFolder, 'bmp')
            listTestFiles2 = list_files(strImageFolder, 'jpg')
            listTestFiles3 = list_files(strImageFolder, 'png')
            listTestFiles = []
            listTestFiles.extend(listTestFiles1)
            listTestFiles.extend(listTestFiles2)
            listTestFiles.extend(listTestFiles3)

        for filenameExt in listTestFiles[3:]:
            filename = os.path.splitext(filenameExt)[0]
            # filenameExt = filename + '.jpg'

            print 'Demo for data/demo/{}'.format(filenameExt)
            plt.close('all')
            if os.path.isfile(os.path.join(strImageFolder, filenameExt)):
                # Load the demo image
                im = cv2.imread(os.path.join(strImageFolder, filenameExt))
                # demo_all(sess, net, im, strPathResult + est_filename)
                if USE_POSEESTIMATE is True:
                    demo_all(sess, net, im, os.path.join(strPathResult, filename + '_est.xml'), extMat, FeatureDB, CoorDB, GeoDB)
                else:
                    demo_all(sess, net, im, os.path.join(strPathResult, filename + '_est.xml'))

                # cv2.waitKey(0)



