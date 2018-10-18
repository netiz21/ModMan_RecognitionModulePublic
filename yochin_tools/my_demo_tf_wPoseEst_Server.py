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
from PoseEst.Function_Pose_v3 import *
import math

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from xml.etree.ElementTree import Element, dump, SubElement, ElementTree
from datetime import datetime

# socket
import socket
import errno

# debugging
import logging
logging.basicConfig(level = logging.INFO)

import select
from time import sleep

# realsense
# import pyrealsense as pyrs

CLASSES = yo_network_info.CLASSES
Candidate_CLASSES = yo_network_info.Candidate_CLASSES
NUM_CLASSES = yo_network_info.NUM_CLASSES

CONF_THRESH = yo_network_info.DETECTION_TH
NMS_THRESH = 0.3

global_do_send_msg = False
global_msg = {'msg': [], 'who': ''}

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

def myimshow(strname, image):
    if image.shape[0] > 600:
        im_resized = cv2.resize(image, None, fx=0.5, fy=0.5)
        cv2.imshow(strname, im_resized)
        cv2.waitKey(10)
    else:
        cv2.imshow(strname, image)
        cv2.waitKey(10)

def demo_all(sess, snet, im_org, strEstPathname, extMat=None, FeatureDB=None, CoorDB=None, GeoDB=None):
    # scalefactor = 300. / float(min(im.shape[0], im.shape[1]))
    # tw = int(im.shape[1] * scalefactor)
    # th = int(im.shape[0] * scalefactor)
    # im = cv2.resize(im, (tw, th))

    ret_list_forKIST = []
    ret_list_BB = []

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im_org)
    timer.toc()

    fontFace = cv2.FONT_HERSHEY_PLAIN
    fontScale = 2
    fontThickness = 2

    if len(strEstPathname) > 0:
        tag_anno = Element('annotation')

    im = im_org.copy()

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

                        print('bbox:')
                        print(bbox)

                        init_coord = np.array([cropbox_lx,  cropbox_ly, 0])    # init_coord[x, y, -], lefttop_point
                        rmat, tvec = PoseEstimate(cropimg, FeatureDB2, CoorDB2, extMat, init_coord)

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

    myimshow('display', im)    

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

class ThreadedClient(object):
    def __init__(self):
        self.owner = '_'
        self.extMat = np.zeros((3, 3))
        self.IMG_WIDTH = 0
        self.IMG_HEIGHT = 0
    def parsing_cmd(self, cmd_raw):
        ret = 0
        buf_size = 3    # 'MME'
        buf_size_intr_params = 0
        buf_size_PadInfo = 0
        buf_size_RTInfo = 0

        cmd = '%c%c'%(struct.unpack('c', cmd_raw[0])[0], struct.unpack('c', cmd_raw[1])[0])

        size_iPad_image = 2224 * 1668 * 3
        size_sr300_image = 640 * 480 * 3

        if len(cmd) == 2:
            if cmd == 'ec': # etri + calibration
                ret = 1
                buf_size_intr_params = 4 * 4
                buf_size_PadInfo = 4 * 12
                buf_size_RTInfo = 0
                buf_size = buf_size + size_iPad_image
            elif cmd == 'er':
                ret = 1
                buf_size_intr_params = 4 * 4
                buf_size_PadInfo = 4 * 12
                buf_size_RTInfo = 0
                buf_size = buf_size + size_iPad_image
            elif cmd == 'ed':
                ret = 1
                buf_size_PadInfo = 4 * 12
                buf_size_RTInfo = 4 * 12
                buf_size = buf_size + 0
            elif cmd == 'ks':
                ret = 1
                buf_size = buf_size + size_sr300_image

        return ret, cmd, buf_size_intr_params, buf_size_PadInfo, buf_size_RTInfo, buf_size

    def listenToClient(self, clientSocket, address_tuple):
        global global_do_send_msg
        global global_msg

        address = address_tuple[0]
        print('Thread start: %s'%address)

        if address == yo_network_info.KIST_STATIC_IP:
            self.owner = 'k'
            print('client (%s) was set as KIST'%address)
        elif address == yo_network_info.ETRI_STATIC_IP:
            self.owner = 'e'
            print('client (%s) was set as ETRI'%address)

        CONNECTED = True
        while CONNECTED == True:  # do image -> get info without interception.
            '''
            Receiving part
            '''
            ing_rcv = False
            len_rcv = 0
            len_rcv_info = 0

            do_capture_success = False

            # get a one image
            NET_BUFSIZE = 3 + 2     # MMS + cmd(2 bytes)
            cmd = '  '
            info_padPosition = np.zeros((12))
            info_RT = np.zeros((12))

            print('Server (%s, %c): connected' % (address, self.owner))

            while CONNECTED == True:
                try:
                    do_read = False

                    timeout_in_seconds = 0.5
                    clientSocket.setblocking(0)

                    while True:
                        try:
                            r, _, _ = select.select([clientSocket], [], [], timeout_in_seconds)
                            do_read = bool(r)
                        except socket.Timeouterror:
                            pass

                        if do_read == True:
                            print('Server (%s, %c): try to receive'%(address, self.owner))
                            data = clientSocket.recv(NET_BUFSIZE)

                            break
                        else:
                            # check there is a data to send
                            if global_do_send_msg == True:
                                if global_msg['who'] == self.owner:
                                    print('Server (%s, %c): This is my message'%(address, self.owner))

                                    # send all in one time
                                    # clientSocket.send(global_msg['msg'])
                                    # print('Server (%s, %c): send to the client'%(address, self.owner))

                                    LEN_CHUNK = yo_network_info.LEN_CHUNK

                                    len_send = 0
                                    for ipack in xrange(0, len(global_msg['msg']), LEN_CHUNK):
                                        len_send = len_send + clientSocket.send(global_msg['msg'][ipack: ipack + LEN_CHUNK])
                                        # print('Server (%s, %c): msg sent: len %d, %d' % (
                                        # address, self.owner, len(global_msg['msg']), len_send))
                                        sleep(0.004)


                                    print('Server (%s, %c): msg sent: len %d, %d' % (address, self.owner, len(global_msg['msg']), len_send))

                                    global_msg = {'msg': [], 'who': ''}
                                    global_do_send_msg = False



                except socket.error, socket.e:
                    if isinstance(socket.e.args, tuple):
                        print("Server (%s, %c): errno is"%(address, self.owner))
                        if socket.e.errno == errno.EPIPE:
                            # remote peer disconnected
                            print("Server (%s, %c): detected remote disconnect"%(address, self.owner))
                        elif socket.e.errno == errno.ECONNRESET:
                            print('Server (%s, %c): disconnected'%(address, self.owner))
                        else:
                            # determine and handle different error
                            print("socket error - type1", socket.e)
                            print('Server (%s, %c): maybe not ready to receive.: '%(address, self.owner))
                    else:
                        print "socket error -type2", socket.e

                    CONNECTED = False
                    clientSocket.close()

                if len(data) == 0:
                    print('Server (%s, %c): we guess the client is disconnected.'%(address, self.owner))
                    CONNECTED = False
                    clientSocket.close()

                if ing_rcv == False and 'MMS' in data:  # first receive - resolve - command, params, then image
                    ing_rcv = True

                    index_start = data.index('MMS')

                    suc_cmd, cmd, buf_size_params, buf_size_PadInfo, buf_size_RTinfo, buf_size = self.parsing_cmd(data[index_start + 3:])

                    if suc_cmd == 0:
                        print('Server (%s, %c): client sends incorrect command'%(address, self.owner))
                        CONNECTED = False
                        clientSocket.close()
                    else:
                        self.owner = cmd[0]
                        NET_BUFSIZE = buf_size
                        print('Server (%s, %c): client sends command [%s]' % (address, self.owner, cmd))

                    if cmd[0] == 'e' and buf_size_params > 0: # update intrinsic params.
                        # get more 4 * 4 bytes and decode them
                        data_params = clientSocket.recv(4 * 4)

                        fx = struct.unpack('f', data_params[:4])[0]  # fx
                        fy = struct.unpack('f', data_params[4:8])[0]  # fy
                        px = struct.unpack('f', data_params[8:12])[0]  # px
                        py = struct.unpack('f', data_params[12:])[0]  # py

                        if USE_REDUCED_IPAD_RESOL == True:
                            px = px * 0.2878
                            py = py * 0.2878

                        print('Set intrinsic param: fxfy (%f, %f), pxpy (%f, %f)' % (fx, fy, px, py))

                        self.extMat[0, 0] = fy  # fy
                        self.extMat[1, 1] = fx  # fx
                        self.extMat[0, 2] = py  # py
                        self.extMat[1, 2] = px  # px
                        self.extMat[2, 2] = 1

                    elif cmd == 'ks':
                        self.extMat = getCamIntParams('SR300')

                    if cmd[0] == 'e':
                        # iPad original resolution
                        self.IMG_WIDTH = 2224
                        self.IMG_HEIGHT = 1668
                        if USE_REDUCED_IPAD_RESOL == True:
                            # iPad reduced resolution
                            self.IMG_WIDTH = 640
                            self.IMG_HEIGHT = 480
                            print('iPad reduced resolution (640x480) applied')
                    elif cmd[0] == 'k':
                        self.IMG_WIDTH = 640
                        self.IMG_HEIGHT = 480

                    if buf_size_PadInfo > 0:
                        # get more 12 * 4 bytes and decode them
                        data_dummy = clientSocket.recv(12 * 4)

                        for i in range(12):
                            info_padPosition[i] = struct.unpack('f', data_dummy[i * 4:i * 4 + 4])[0]

                        print('Pad Pos. : ', info_padPosition)

                    if buf_size_RTinfo > 0:
                        # get more 12 * 4 bytes and decode them
                        data_dummy = clientSocket.recv(12 * 4)

                        for i in range(12):
                            info_RT[i] = struct.unpack('f', data_dummy[i*4:i*4+4])[0]  # fx

                        print('Dst Pos.: ', info_RT)

                    fid_bin = open('./skku_img.bin', 'wb')

                    len_rcv = 0

                elif ing_rcv == True:  # receive again.
                    if ('MME' in data) and (data.index('MME') + len_rcv == buf_size-3):
                        ing_rcv = False
                        index_end = data.index('MME')
                        fid_bin.write(data[:index_end])
                        fid_bin.close()

                        len_rcv = len_rcv + len(data[:index_end])
                        print('final: %d == %d' % (buf_size-3, len_rcv))

                        len_rcv = 0

                        self.img = np.fromfile('./skku_img.bin', dtype='uint8')

                        if len(self.img) != self.IMG_HEIGHT * self.IMG_WIDTH * 3:
                            print('captured image size is not predefined size')
                            do_capture_success = False
                        else:
                            self.img = self.img.reshape(self.IMG_HEIGHT, self.IMG_WIDTH, 3)
                            do_capture_success = True

                        break
                    else:
                        fid_bin.write(data)
                        len_rcv = len_rcv + len(data)

                        # print(data)
                # print('intermediate: %d == %d' % (AR_NET_BUFSIZE, len_rcv))

            print('Server (%s, %c): received data completely'%(address, self.owner))


            # do ETRI job
            if do_capture_success is True:
                myimshow('display', self.img)

                if DO_WRITE_RESULT_IMG == True:
                    timestamp = get_curtime_as_string()
                    cv2.imwrite('../debug_dropbox_upload/%s.png' % timestamp, self.img)
                    print('captured %s image', timestamp)

                cv2.imwrite('./debug_img.png', self.img)

                if cmd == 'er' or cmd == 'ks':
                    print('Server (%s, %c): process image...' % (address, self.owner))
                    if USE_POSEESTIMATE is True:
                        result_img, list_objs_forAR, _ = demo_all(sess, net, self.img, '', self.extMat, FeatureDB, CoorDB, GeoDB)
                    else:
                        result_img, list_objs_forAR, _ = demo_all(sess, net, self.img, '')

                if DO_WRITE_RESULT_IMG == True:
                    cv2.imwrite('../debug_dropbox_upload/%s_est.png' % timestamp, result_img)

                    fid_info = open('../debug_dropbox_upload/%s_est.txt' % timestamp, 'w')
                    print >> fid_info, list_objs_forAR
                    fid_info.close()

                    print('save dummy image as a file')
            else:
                print('no frame\n')

            # send to kist
            # 4 byte: num of object (int)
            # 1 byte: obj ID (char)
            # 4 * (9 + 3) = 48 bytes = rot + trans mat
            # 4 * 2 = 8 bytes = x, y

            '''
            Sending part
            '''
            if CONNECTED == True:
                if cmd == 'ec':
                    msg = 'MMS'
                    msg = addCommand(msg, cmd)
                    msg = addPadRTInfo(msg, info_padPosition, floatType='double')
                    msg = addImage(msg, self.img)
                    msg = msg + 'MME'

                    global_msg['msg'] = copy.copy(msg)
                    global_msg['who'] = 'k'
                    global_do_send_msg = True
                    print('Server (%s, %c): set global msg _%s_'%(address, self.owner, msg[:5]))
                elif cmd == 'ed':
                    msg = 'MMS'
                    msg = addCommand(msg, cmd)
                    msg = addPadRTInfo(msg, info_padPosition, floatType='double')
                    msg = addPadRTInfo(msg, info_RT, floatType='double')
                    msg = msg + 'MME'

                    global_msg['msg'] = copy.copy(msg)
                    global_msg['who'] = 'k'
                    global_do_send_msg = True
                    print('Server (%s, %c): set global msg _%s_' % (address, self.owner, msg))
                    print('Server (%s, %c): (cont.) ' % (address, self.owner), msg[5:5 + 24 * 8])
                else:
                    # message to loopback.
                    msg = 'MMS'
                    msg = addCommand(msg, cmd)

                    if cmd == 'er':
                        msg = addListObjs(msg, list_objs_forAR, floatType='float')
                    elif cmd =='ks':
                        msg = addListObjs(msg, list_objs_forAR, floatType='double')
                    msg = msg + 'MME'

                    # for list_objs
                    clientSocket.send(msg)
                    print('Server (%s, %c): send _%s_ to the client' % (address, self.owner, msg))

                    if cmd == 'er':
                        msg = 'MMS'
                        msg = addCommand(msg, cmd)
                        # msg = addImage(msg, self.img)
                        msg = addPadRTInfo(msg, info_padPosition, floatType='double')
                        msg = addListObjs(msg, list_objs_forAR, floatType='double')
                        msg = msg + 'MME'

                        global_msg['msg'] = copy.copy(msg)
                        global_msg['who'] = 'k'
                        global_do_send_msg = True

                        print('Server (%s, %c): set global msg '%(address, self.owner))

# https://stackoverflow.com/questions/23828264/how-to-make-a-simple-multithreaded-socket-server-in-python-that-remembers-client
class ThreadedServer(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))

    def listen(self):
        self.sock.listen(5)     # accept max. 5 clients
        while True:
            client, address = self.sock.accept()
            client.settimeout(30*60)               # if there is no reaction for 30*60s, it will disconnect and close.

            clientInst = ThreadedClient()

            threading.Thread(target = clientInst.listenToClient,\
                             args = (client, address)\
                             ).start()


def addPadRTInfo(msg, padinfo, floatType):
    if floatType == 'float':
        for i in range(12):
            msg = msg + struct.pack('f', padinfo[i])
    elif floatType == 'double':
        for i in range(12):
            msg = msg + struct.pack('d', padinfo[i])
    else:
        raise AssertionError('type is incorrect!!!')

    print(padinfo)

    return msg

def addImage(msg, image):
    # Image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(3):
                str_c = '%c' % image[i, j, c]
                # msg = msg + struct.pack('c', str_c.encode('ascii'))
                msg = msg + struct.pack('c', str_c)

    return msg


def addListObjs(msg, listObjs, floatType):
    # num_of_objects: int x 1
    msg = msg + struct.pack('i', len(listObjs))  # int
    print('num of objs: %d' % len(listObjs))

    for obj_AR in listObjs:
        msg = msg + struct.pack('c', obj_AR['object'][0])
        print('obj name: %c' % obj_AR['object'][0])

        if floatType == 'float':
            for j_RMat in range(0, 3):
                for i_RMat in range(0, 3):
                    msg = msg + struct.pack('f', obj_AR['RMat'][i_RMat][j_RMat])
            print('RMat:')
            print(obj_AR['RMat'])

            for j_TVec in range(0, 3):
                msg = msg + struct.pack('f', obj_AR['TVec'][j_TVec][0])
            print('TVec:')
            print(obj_AR['TVec'])
        elif floatType == 'double':
            for j_RMat in range(0, 3):
                for i_RMat in range(0, 3):
                    msg = msg + struct.pack('d', obj_AR['RMat'][i_RMat][j_RMat])
            print('RMat:')
            print(obj_AR['RMat'])

            for j_TVec in range(0, 3):
                msg = msg + struct.pack('d', obj_AR['TVec'][j_TVec][0])
            print('TVec:')
            print(obj_AR['TVec'])
        else:
            raise AssertionError('type is incorrect!!!')

        msg = msg + struct.pack('i', int(obj_AR['left'])) \
              + struct.pack('i', int(obj_AR['top'])) \
              + struct.pack('i', int(obj_AR['right'])) \
              + struct.pack('i', int(obj_AR['bottom']))

        print('left: %d' % (int(obj_AR['left'])))
        print('top: %d' % (int(obj_AR['top'])))
        print('right: %d' % (int(obj_AR['right'])))
        print('bottom: %d' % (int(obj_AR['bottom'])))

    return msg


def addCommand(msg, cmd):
    # command: 2bytes
    msg = msg + struct.pack('c', cmd[0])
    msg = msg + struct.pack('c', cmd[1])

    return msg

def setTestCFG(cfg):
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
    # cfg.TEST.MAX_SIZE = 2000

    return cfg

# variable to communicate between threads.
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    '''
    Settings
    '''
    USE_POSEESTIMATE = True
    USE_REDUCED_IPAD_RESOL = True

    Svr_IP = yo_network_info.SERVER_IP
    Svr_PORT = yo_network_info.SERVER_PORT

    DO_WRITE_RESULT_AVI = False
    name_output_avi = 'output.avi'

    DO_WRITE_RESULT_IMG = False     # IMG name will be the current time.

    # for test
    cfg = setTestCFG(cfg)

    avgwindow = 0   # parameter to show pose estimation in stable


    '''
    Write Result
    '''
    if DO_WRITE_RESULT_AVI == True:
        # Define the codec and create VideoWriter object
        frame_width = 640
        frame_height = 480
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

    # working as a server
    '''
    Server Info
    '''
    print('Server: waiting of client connection')
    ThreadedServer(Svr_IP, Svr_PORT).listen()
