# coding=utf-8
# #!/usr/bin/env python

# default library
import os
import sys
import struct
import copy

import yo_network_info
sys.path.append(os.path.join(yo_network_info.PATH_BASE, 'lib'))
# sys.path.append('/usr/lib/python2.7/dist-packages')

# Faster-RCNN_TF
from networks.factory import get_network
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

# ModMan module & Pose estimation
from PoseEst.Function_Pose_v1 import *
import math

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from xml.etree.ElementTree import Element, dump, SubElement, ElementTree
from datetime import datetime

# socket
from socket import *
import errno

# debugging
import logging
logging.basicConfig(level = logging.INFO)

# realsense
import pyrealsense as pyrs

CLASSES = yo_network_info.CLASSES
Candidate_CLASSES = yo_network_info.Candidate_CLASSES
NUM_CLASSES = yo_network_info.NUM_CLASSES

CONF_THRESH = 0.6
NMS_THRESH = 0.3

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

def demo(sess, net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    # print ('Detection took {:.3f}s for '
    #        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

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

def demo_all(sess, snet, im, strEstPathname, extMat=None, FeatureDB=None, CoorDB=None, GeoDB=None):
    # scalefactor = 300. / float(min(im.shape[0], im.shape[1]))
    # tw = int(im.shape[1] * scalefactor)
    # th = int(im.shape[0] * scalefactor)
    # im = cv2.resize(im, (tw, th))

    ret_list_forKIST = []
    ret_list_BB = []

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    # print ('\t\t\t\tDetection took {:.3f}s for '
    #        '{:d} object proposals').format(timer.total_time, boxes.shape[0])

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

                    cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (int(bbox[0]), int(bbox[1] - 2)), fontFace, fontScale, fontColor, thickness = fontThickness)

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




    cv2.imshow('display', im)
    cv2.waitKey(10)

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


import threading

# https://stackoverflow.com/questions/23828264/how-to-make-a-simple-multithreaded-socket-server-in-python-that-remembers-client
class ThreadedServer(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))

    def listen(self):
        self.sock.listen(5)
        while True:
            client, address = self.sock.accept()
            client.settimeout(60)
            threading.Thread(target = self.listenToClient,args = (client, address)).start()

    def listenToClient(self, client, address):
        size = 1024
        while True:
            try:
                data = client.recv(size)
                if data:
                    # Set the response to echo back the recieved data
                    response = data
                    client.send(response)
                else:
                    raise error('Client disconnected')
            except:
                client.close()
                return False





if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    '''
    Settings
    '''
    INPUT_TYPE = 8      # 0: WebCamera,
                        # 3: Video
                        # 6: Realsense Camera

                        # 5: working as server for IPad,
                        # 7: working as server for SR300
    USE_POSEESTIMATE = True

    # Svr_IP = '129.254.87.77'
    Svr_IP = socket.gethostname()
    Svr_PORT = 8020

    if USE_POSEESTIMATE == True:    # Cam Intrinsic Params Settings
        extMat = getCamIntParams('client')

    DO_WRITE_RESULT_AVI = False
    name_output_avi = 'output.avi'

    DO_WRITE_RESULT_IMG = True     # IMG name is the current timeshot.

    # for test
    cfg.NCLASSES = NUM_CLASSES
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.BBOX_REG = True  # Use BBox regressor
    # # Scales to use during testing (can list multiple scales)
    # # Each scale is the pixel size of an image's shortest side
    cfg.TEST.SCALES = (1600,)  # only one for test  ,480, 576, 688, 864, 1200
    cfg.TEST.RPN_NMS_THRESH = 0.7

    ## Number of top scoring boxes to keep before apply NMS to RPN proposals
    # cfg.TEST.RPN_PRE_NMS_TOP_N = 6000
    cfg.TEST.RPN_PRE_NMS_TOP_N = 12000
    ## Number of top scoring boxes to keep after applying NMS to RPN proposals
    # cfg.TEST.RPN_POST_NMS_TOP_N = 300
    cfg.TEST.RPN_POST_NMS_TOP_N = 2000
    CONF_THRESH = 0.8
    # cfg.TEST.MAX_SIZE = 2000

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
        cap = cv2.VideoCapture(0)
        # cap.set(3, 640*2)
        # cap.set(4, 480*2)
    elif INPUT_TYPE == 6:
        serv = pyrs.Service()
        # pyrs.start()

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

    if INPUT_TYPE == 0 or INPUT_TYPE == 3:
        while (True):
            ret, frame = cap.read()

            cv2.imwrite('./debug_img.png', frame)

            if ret is True:
                if USE_POSEESTIMATE is True:
                    im, list_objs_forKIST, _ = demo_all(sess, net, frame, '', extMat, FeatureDB, CoorDB, GeoDB)
                else:
                    im, _, _ = demo_all(sess, net, frame, '')

                if DO_WRITE_RESULT_AVI == True:
                    print(im.shape)
                    outavi.write(im)
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
            elif input_key == ord('c'):
                timestamp = '%d_%d_%d_%d_%d_%d_%d'%(datetime.now().year, datetime.now().month, datetime.now().day, datetime.now().hour, datetime.now().minute, datetime.now().second, datetime.now().microsecond)
                cv2.imwrite('./%s.png'%timestamp, frame)
                print('captured %s image', timestamp)
            elif input_key == ord('p'):
                USE_POSEESTIMATE = 1 - USE_POSEESTIMATE
                print('USE_POSEESTIMATE: %d'%(USE_POSEESTIMATE))

        if DO_WRITE_RESULT_AVI == True:
            outavi.release()
            print('outavi is released')
        cap.release()
        cv2.destroyAllWindows()
    elif INPUT_TYPE == 8:
        # working as a server
        '''
        Server Info
        '''
        GET_PARAMS_MORE = False
        if np.sum(np.abs(extMat)) == 0:     # if extMat has all zeros, then get intrisic params from client PC.
            GET_PARAMS_MORE = True

        print('Server: waiting of client connection')
        ThreadedServer(Svr_IP, Svr_PORT).listen()

        # Svr_serverSocket.listen(5)




        '''
        Data Info - Length for data
        '''
        IPAD_IMG_WIDTH = 2224
        IPAD_IMG_HEIGHT = 1668
        IPAD_NET_BUFSIZE = (IPAD_IMG_WIDTH * IPAD_IMG_HEIGHT * 3 + 6)





    elif INPUT_TYPE == 5:
        # working as a server
        # '''
        # Server Info
        # '''
        # AR_ADDR = (AR_IP, AR_PORT)
        # AR_serverSocket = socket(AF_INET, SOCK_STREAM)
        # AR_serverSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        # AR_serverSocket.bind(AR_ADDR)

        # GET_PARAMS_MORE = False
        # if np.sum(np.abs(extMat)) == 0:     # if extMat has all zeros, then get intrisic params from client PC.
        #     GET_PARAMS_MORE = True

        '''
        Data Info - Length for data
        '''
        IPAD_IMG_WIDTH = 2224
        IPAD_IMG_HEIGHT = 1668
        IPAD_NET_BUFSIZE = (IPAD_IMG_WIDTH * IPAD_IMG_HEIGHT * 3 + 6)

        if GET_PARAMS_MORE == True:
            # fx, fy, cx, cy = 4 float numbers = 4 * 4 Bytes = 16 Bytes
            LEN_PARAM_INFO = 4*4
            IPAD_NET_BUFSIZE = IPAD_NET_BUFSIZE + LEN_PARAM_INFO

        while True: # disconnect -> wait new connection
            print('Server: waiting of client connection')
            Svr_serverSocket.listen(5)
            AR_serverSocket.settimeout(5)  # sec.

            try:
                AR_clientSocket, clientAddr = AR_serverSocket.accept()
            except timeout:
                print('Server: timeout. wait the client, again')
                continue

            print('Server: connected to the client (%s:%s)' % clientAddr)

            CONNECTED = True
            while CONNECTED == True: # do image -> get info without interception.
                n_stacked_result = 0
                while CONNECTED == True:
                    ing_rcv = False
                    len_rcv = 0
                    len_rcv_info = 0

                    do_capture_success = False

                    # get a one image
                    while CONNECTED == True:
                        try:
                            print('Server: try to receive')
                            data = AR_clientSocket.recv(AR_NET_BUFSIZE)
                            # print('received %d'%(len(data)))
                        except error, e:
                            if isinstance(e.args, tuple):
                                print "Server: errno is %d" % e[0]
                                if e.errno == errno.EPIPE:
                                    # remote peer disconnected
                                    print "Server: detected remote disconnect"
                                elif e.errno == errno.ECONNRESET:
                                    print 'Server: disconnected'
                                else:
                                    # determine and handle different error
                                    print "socket error ", e
                                    pass
                            else:
                                print "socket error ", e

                            CONNECTED = False
                            AR_clientSocket.close()

                        if len(data) == 0:
                            print('Server: we guess the client is disconnected.')
                            CONNECTED = False
                            AR_clientSocket.close()

                        if ing_rcv == False and 'MMS' in data:  # first receive
                            ing_rcv = True

                            index_start = data.index('MMS')

                            if GET_PARAMS_MORE == True and len_rcv_info == 0:
                                data_params = copy.copy(data[index_start + 3: index_start + 3 + LEN_PARAM_INFO])

                                fx = struct.unpack('f', data_params[:4])[0]        # fx
                                fy = struct.unpack('f', data_params[4:8])[0]       # fy
                                px = struct.unpack('f', data_params[8:12])[0]      # px
                                py = struct.unpack('f', data_params[12:])[0]       # py

                                # KinectV2
                                print('Set intrinsic param: fxfy (%f, %f), pxpy (%f, %f)'%(fx, fy, px, py))
                                extMat[0, 0] = fy  # fy
                                extMat[1, 1] = fx  # fx
                                extMat[0, 2] = py  # py
                                extMat[1, 2] = px  # px
                                extMat[2, 2] = 1


                                len_rcv_info = LEN_PARAM_INFO
                                index_start = index_start + LEN_PARAM_INFO

                            fid_bin = open('./skku_img.bin', 'wb')

                            if 'MME' not in data:               # not include end point
                                fid_bin.write(data[index_start + 3:])
                                len_rcv = len_rcv + len(data[index_start + 3:])
                            else:                               # if include end point
                                index_end = data.index('MME')
                                fid_bin.write(data[index_start + 3:index_end])
                                fid_bin.close()
                                print('final: %d == %d'%(AR_NET_BUFSIZE-6-LEN_PARAM_INFO, len_rcv))

                                len_rcv = 0

                                img = np.fromfile('./skku_img.bin', dtype='uint8')

                                if len(img) != AR_IMG_HEIGHT * AR_IMG_WIDTH * 3:
                                    print('captured image size is not same with predefined')
                                    do_capture_success = False
                                else:
                                    img = img.reshape(AR_IMG_HEIGHT, AR_IMG_WIDTH, 3)
                                    do_capture_success = True
                                break
                        elif ing_rcv == True:                   # receive again.
                            if ('MME' in data) and (data.index('MME') + len_rcv == AR_IMG_WIDTH * AR_IMG_HEIGHT * 3):
                                ing_rcv = False
                                index_end = data.index('MME')
                                fid_bin.write(data[:index_end])
                                fid_bin.close()

                                len_rcv = len_rcv + len(data[:index_end])
                                print('final: %d == %d' % (AR_NET_BUFSIZE-6, len_rcv))

                                len_rcv = 0

                                img = np.fromfile('./skku_img.bin', dtype='uint8')

                                if len(img) != AR_IMG_HEIGHT * AR_IMG_WIDTH * 3:
                                    print('captured image size is not predefined size')
                                    do_capture_success = False
                                else:
                                    img = img.reshape(AR_IMG_HEIGHT, AR_IMG_WIDTH, 3)
                                    do_capture_success = True

                                break
                            else:
                                fid_bin.write(data)
                                len_rcv = len_rcv + len(data)

                                # print(data)
                        # print('intermediate: %d == %d' % (AR_NET_BUFSIZE, len_rcv))

                    print('Server: received image')

                    # # do ETRI job
                    # # # temp
                    # img = np.fromfile('./skku_img.bin', dtype='uint8')
                    # img = img.reshape(AR_IMG_WIDTH, AR_IMG_HEIGHT, 3)
                    # do_capture_success = True

                    if do_capture_success is True:
                        # img = np.array(np.rot90(img))
                        # img = img.copy()


                        cv2.imshow('display', img)
                        cv2.waitKey(10)

                        if DO_WRITE_RESULT_IMG == True:
                            timestamp = get_curtime_as_string()
                            cv2.imwrite('../debug_dropbox_upload/%s.png'%timestamp, img)
                            print('captured %s image', timestamp)

                        cv2.imwrite('./debug_img.png', img)

                        if USE_POSEESTIMATE is True:
                            img, list_objs_forAR, _ = demo_all(sess, net, img, '', extMat, FeatureDB, CoorDB, GeoDB)
                        else:
                            demo_all(sess, net, img, '')

                        if DO_WRITE_RESULT_IMG == True:
                            cv2.imwrite('../debug_dropbox_upload/%s_est.png' % timestamp, img)

                            fid_info = open('../debug_dropbox_upload/%s_est.txt' % timestamp, 'w')
                            print >> fid_info, list_objs_forAR
                            fid_info.close()

                            print('save dummy image as a file')


                        # # temp - start
                        # list_objs_forAR = []
                        # obj_info = {'object': 'Hello', 'score': 100.,
                        #             'RMat': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                        #             'TVec': np.array([[10], [20], [30]]),
                        #             'x_center': 100, 'y_center': 200}
                        # list_objs_forAR.append(obj_info)
                        # # temp - end
                    else:
                        print('no frame\n')

                    do_capture_success = False

                    if CONNECTED == False:
                        break

                    # cv2.imshow('frame', frame)

                    input_key = cv2.waitKey(10)

                    if len(list_objs_forAR) > 0:
                        n_stacked_result = n_stacked_result + 1

                    if n_stacked_result >= avgwindow:
                        break

                # send to kist
                # 4 byte: num of object (int)
                # 1 byte: obj ID (char)
                # 4 * (9 + 3) = 48 bytes = rot + trans mat
                # 4 * 2 = 8 bytes = x, y

                if CONNECTED == True:
                    msg = 'MMS'
                    msg = msg + struct.pack('i', len(list_objs_forAR))  # int
                    print('num of objs: %d' % len(list_objs_forAR))

                    for obj_AR in list_objs_forAR:
                        msg = msg + struct.pack('c', obj_AR['object'][0])
                        print('obj name: %c' % obj_AR['object'][0])

                        for j_RMat in range(0, 3):
                            for i_RMat in range(0, 3):
                                msg = msg + struct.pack('f', obj_AR['RMat'][i_RMat][j_RMat])
                        print('RMat:')
                        print(obj_AR['RMat'])

                        for j_TVec in range(0, 3):
                            msg = msg + struct.pack('f', obj_AR['TVec'][j_TVec][0])
                        print('TVec:')
                        print(obj_AR['TVec'])

                        msg = msg + struct.pack('i', int(obj_AR['x_center'])) + struct.pack('i',
                                                                                              int(obj_AR['y_center']))
                        print('x_center: %d' % (int(obj_AR['x_center'])))
                        print('y_center: %d' % (int(obj_AR['y_center'])))

                    msg = msg + 'MME'

                    # for list_objs
                    AR_clientSocket.send(msg)
                    print('send _%s_ to AR server' % (msg))
    elif INPUT_TYPE == 7:
        # working as a server for SR300
        '''
        Server Info
        '''
        AR_ADDR = (AR_IP, AR_PORT)
        AR_serverSocket = socket(AF_INET, SOCK_STREAM)
        AR_serverSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        AR_serverSocket.bind(AR_ADDR)

        extMat = getCamIntParams('SR300')

        GET_PARAMS_MORE = False
        if np.sum(np.abs(extMat)) == 0:     # if extMat has all zeros, then get intrisic params from client PC.
            GET_PARAMS_MORE = True

        '''
        Data Info
        '''
        AR_IMG_WIDTH = 640
        AR_IMG_HEIGHT = 480
        AR_NET_BUFSIZE = (AR_IMG_WIDTH * AR_IMG_HEIGHT * 3 + 6)

        LEN_PARAM_INFO = 0
        if GET_PARAMS_MORE == True:
            # fx, fy, cx, cy = 4 float numbers = 4 * 4 Bytes = 16 Bytes
            LEN_PARAM_INFO = 4*4
            AR_NET_BUFSIZE = AR_NET_BUFSIZE + LEN_PARAM_INFO

        while True: # disconnect -> wait new connection
            print('Server: waiting of client connection')
            AR_serverSocket.listen(5)
            AR_serverSocket.settimeout(5)  # sec.

            try:
                AR_clientSocket, clientAddr = AR_serverSocket.accept()
            except timeout:
                print('Server: timeout. wait the client, again')
                continue

            print('Server: connected to the client (%s:%s)' % clientAddr)

            CONNECTED = True
            while CONNECTED == True: # do image -> get info without interception.
                n_stacked_result = 0
                while CONNECTED == True:
                    ing_rcv = False
                    len_rcv = 0
                    len_rcv_info = 0

                    do_capture_success = False

                    # get a one image
                    while CONNECTED == True:
                        try:
                            print('Server: try to receive')
                            data = AR_clientSocket.recv(AR_NET_BUFSIZE)
                            # print('received %d'%(len(data)))
                        except error, e:
                            if isinstance(e.args, tuple):
                                print "Server: errno is %d" % e[0]
                                if e.errno == errno.EPIPE:
                                    # remote peer disconnected
                                    print "Server: detected remote disconnect"
                                elif e.errno == errno.ECONNRESET:
                                    print 'Server: disconnected'
                                else:
                                    # determine and handle different error
                                    print "socket error ", e
                                    pass
                            else:
                                print "socket error ", e

                            CONNECTED = False
                            AR_clientSocket.close()

                                # socket.timeout:
                        if len(data) == 0:
                            print('Server: we guess the client is disconnected.')
                            CONNECTED = False
                            AR_clientSocket.close()

                        #     CONNECTED = False
                        #     print('Server: Connection with client is timeout.')
                        #     socket.close()
                        #
                        #     break
                        # except socket.EPIPE:
                        #     CONNECTED = False
                        #     print('Server: Connection with client is disconnected.')
                        #
                        #     break

                        if ing_rcv == False and 'MMS' in data:  # first receive
                            ing_rcv = True

                            index_start = data.index('MMS')

                            if GET_PARAMS_MORE == True and len_rcv_info == 0:
                                data_params = copy.copy(data[index_start + 3: index_start + 3 + LEN_PARAM_INFO])

                                fx = struct.unpack('f', data_params[:4])[0]        # fx
                                fy = struct.unpack('f', data_params[4:8])[0]       # fy
                                px = struct.unpack('f', data_params[8:12])[0]      # px
                                py = struct.unpack('f', data_params[12:])[0]       # py

                                # KinectV2
                                print('Set intrinsic param: fxfy (%f, %f), pxpy (%f, %f)'%(fx, fy, px, py))
                                extMat[0, 0] = fy  # fy
                                extMat[1, 1] = fx  # fx
                                extMat[0, 2] = py  # py
                                extMat[1, 2] = px  # px
                                extMat[2, 2] = 1


                                len_rcv_info = LEN_PARAM_INFO
                                index_start = index_start + LEN_PARAM_INFO

                            fid_bin = open('./skku_img.bin', 'wb')

                            if 'MME' not in data:               # not include end point
                                fid_bin.write(data[index_start + 3:])
                                len_rcv = len_rcv + len(data[index_start + 3:])
                            else:                               # if include end point
                                index_end = data.index('MME')
                                fid_bin.write(data[index_start + 3:index_end])
                                fid_bin.close()
                                print('final: %d == %d'%(AR_NET_BUFSIZE-6-LEN_PARAM_INFO, len_rcv))

                                len_rcv = 0

                                img = np.fromfile('./skku_img.bin', dtype='uint8')

                                if len(img) != AR_IMG_HEIGHT * AR_IMG_WIDTH * 3:
                                    print('captured image size is not same with predefined')
                                    do_capture_success = False
                                else:
                                    img = img.reshape(AR_IMG_HEIGHT, AR_IMG_WIDTH, 3)
                                    do_capture_success = True
                                break
                        elif ing_rcv == True:                   # receive again.
                            if ('MME' in data) and (data.index('MME') + len_rcv == AR_IMG_WIDTH * AR_IMG_HEIGHT * 3):
                                ing_rcv = False
                                index_end = data.index('MME')
                                fid_bin.write(data[:index_end])
                                fid_bin.close()

                                len_rcv = len_rcv + len(data[:index_end])
                                print('final: %d == %d' % (AR_NET_BUFSIZE-6, len_rcv))

                                len_rcv = 0

                                img = np.fromfile('./skku_img.bin', dtype='uint8')

                                if len(img) != AR_IMG_HEIGHT * AR_IMG_WIDTH * 3:
                                    print('captured image size is not predefined size')
                                    do_capture_success = False
                                else:
                                    img = img.reshape(AR_IMG_HEIGHT, AR_IMG_WIDTH, 3)
                                    do_capture_success = True

                                break
                            else:
                                fid_bin.write(data)
                                len_rcv = len_rcv + len(data)

                                # print(data)
                        # print('intermediate: %d == %d' % (AR_NET_BUFSIZE, len_rcv))

                    print('Server: received image')

                    # # do ETRI job
                    # # # temp
                    # img = np.fromfile('./skku_img.bin', dtype='uint8')
                    # img = img.reshape(AR_IMG_WIDTH, AR_IMG_HEIGHT, 3)
                    # do_capture_success = True

                    if do_capture_success is True:
                        # img = np.array(np.rot90(img))
                        # img = img.copy()


                        cv2.imshow('display', img)
                        cv2.waitKey(10)

                        if DO_WRITE_RESULT_IMG == True:
                            timestamp = get_curtime_as_string()
                            cv2.imwrite('../debug_dropbox_upload/%s.png'%timestamp, img)
                            print('captured %s image', timestamp)

                        cv2.imwrite('./debug_img.png', img)

                        if USE_POSEESTIMATE is True:
                            img, list_objs_forAR, _ = demo_all(sess, net, img, '', extMat, FeatureDB, CoorDB, GeoDB)
                        else:
                            demo_all(sess, net, img, '')

                        if DO_WRITE_RESULT_IMG == True:
                            cv2.imwrite('../debug_dropbox_upload/%s_est.png' % timestamp, img)

                            fid_info = open('../debug_dropbox_upload/%s_est.txt' % timestamp, 'w')
                            print >> fid_info, list_objs_forAR
                            fid_info.close()

                            print('save dummy image as a file')


                        # # temp - start
                        # list_objs_forAR = []
                        # obj_info = {'object': 'Hello', 'score': 100.,
                        #             'RMat': np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                        #             'TVec': np.array([[10], [20], [30]]),
                        #             'x_center': 100, 'y_center': 200}
                        # list_objs_forAR.append(obj_info)
                        # # temp - end
                    else:
                        print('no frame\n')

                    do_capture_success = False

                    if CONNECTED == False:
                        break

                    # cv2.imshow('frame', frame)

                    input_key = cv2.waitKey(10)

                    if len(list_objs_forAR) > 0:
                        n_stacked_result = n_stacked_result + 1

                    if n_stacked_result >= avgwindow:
                        break

                # send to kist
                # 4 byte: num of object (int)
                # 1 byte: obj ID (char)
                # 4 * (9 + 3) = 48 bytes = rot + trans mat
                # 4 * 2 = 8 bytes = x, y

                if CONNECTED == True:
                    msg = 'MMS'
                    msg = msg + struct.pack('i', len(list_objs_forAR))  # int
                    print('num of objs: %d' % len(list_objs_forAR))

                    for obj_AR in list_objs_forAR:
                        msg = msg + struct.pack('c', obj_AR['object'][0])
                        print('obj name: %c' % obj_AR['object'][0])

                        for j_RMat in range(0, 3):
                            for i_RMat in range(0, 3):
                                msg = msg + struct.pack('f', obj_AR['RMat'][i_RMat][j_RMat])
                        print('RMat:')
                        print(obj_AR['RMat'])

                        for j_TVec in range(0, 3):
                            msg = msg + struct.pack('f', obj_AR['TVec'][j_TVec][0])
                        print('TVec:')
                        print(obj_AR['TVec'])

                        msg = msg + struct.pack('i', int(obj_AR['left'])) \
                                  + struct.pack('i', int(obj_AR['top'])) \
                                  + struct.pack('i', int(obj_AR['right'])) \
                                  + struct.pack('i', int(obj_AR['bottom']))

                        print('left: %d' % (int(obj_AR['left'])))
                        print('top: %d' % (int(obj_AR['top'])))
                        print('right: %d' % (int(obj_AR['right'])))
                        print('bottom: %d' % (int(obj_AR['bottom'])))

                    msg = msg + 'MME'

                    # for list_objs
                    AR_clientSocket.send(msg)
                    print('send _%s_ to AR server' % (msg))
