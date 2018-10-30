# coding=utf-8
__author__ = 'yochin'

import os
import sys
import cv2
import codecs
import copy
import random as rnd
import numpy as np
import xml.etree.ElementTree as ET

def list_files(path, ext):
    filelist = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            if name.endswith(ext):
                filelist.append(name)
    return filelist

if __name__ == '__main__':
    '''
    This is the main function
    '''
    saveBaseFolder = '/home/yochin/Desktop/ModMan_ETRI/data/'
    savedFolder_Infos = saveBaseFolder + 'Annotations'
    savedFolder_Images = saveBaseFolder + 'Images'
    savedFolder_ImageSets = saveBaseFolder + 'ImageSets'
    do_show_image = False
    do_check_outofbound = True
    xml_zerobase = False

    listFiles = codecs.open(savedFolder_ImageSets + '/train.txt').read().split('\n')

    for ifile, filename in enumerate(listFiles):
        if len(filename) > 0:

            if ifile >= 0 and ifile <= 1000000000000000:
                strFullPathImage = savedFolder_Images + '/' + filename + '.png'

                if os.path.isfile(strFullPathImage) is False:
                    strFullPathImage = savedFolder_Images + '/' + filename + '.jpg'

                im = cv2.imread(strFullPathImage)

                # text file type
                # strFullPathAnno = savedFolder_Infos + '/' + filename + '.txt'
                # listInfos = codecs.open(strFullPathAnno).read().split('\n')
                # for info in listInfos:
                #     if len(info) > 0:
                #         each_info = info.split(' ')
                #         Obj = each_info[0]
                #         ty = int(each_info[1])
                #         tx = int(each_info[2])
                #         ty2 = int(each_info[3])
                #         tx2 = int(each_info[4])
                #
                #         cv2.rectangle(im, (tx, ty), (tx2, ty2), (255,0,0), 2)
                #
                #         font = cv2.FONT_HERSHEY_SIMPLEX
                #         cv2.putText(im, Obj, (tx, ty), font, 1,(255,255,255),2)

                # xml file type
                strFullPathAnno = savedFolder_Infos + '/' + filename + '.xml'
                tree = ET.parse(strFullPathAnno)
                objs = tree.findall('object')
                num_objs = len(objs)  # number of object in one image

                # Load object bounding boxes into a data frame.
                for ix, obj in enumerate(objs):
                    bbox = obj.find('bndbox')
                    # Make pixel indexes 0-based
                    tx = int(bbox.find('xmin').text)
                    ty = int(bbox.find('ymin').text)
                    tx2 = int(bbox.find('xmax').text)
                    ty2 = int(bbox.find('ymax').text)

                    if do_check_outofbound == True:
                        # print('%d: %s file - %d object'%(ifile, filename, ix))

                        if xml_zerobase == False:
                            tx = tx-1
                            ty = ty-1
                            tx2 = tx2-1
                            ty2 = ty2-1

                        assert tx > 0, '%s: left point tx(%d) >= 0'%(filename, tx)
                        assert ty > 0, '%s: top point ty(%d) >= 0'%(filename, ty)
                        assert tx2 < im.shape[1]-1, '%s, right point tx2(%d) <= width(%d)'%(filename, tx2, im.shape[1])
                        assert ty2 < im.shape[0]-1, '%s, bottom point ty2(%d) <= height(%d)'%(filename, ty2, im.shape[0])

                        assert tx < tx2, '%s, left point tx < right point tx2'%(filename)
                        assert ty < ty2, '%s, top point ty < bottom point ty2'%(filename)

                        assert float(im.shape[0])/float(im.shape[1]) < 4, '%s, ratio > 4'%(filename)
                        assert float(im.shape[1])/float(im.shape[0]) < 4, '%s, ratio > 4'%(filename)

                        if np.isnan(tx * ty * tx2 * ty2) is True:
                            print('nan data: %d %d %d %d' % (tx, ty, tx2, ty2))

                    if do_show_image == True:
                        cv2.rectangle(im, (tx, ty), (tx2, ty2), (255, 0, 0), 2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(im, obj.find('name').text, (tx, ty), font, 1,(255,255,255),2)

                if do_show_image == True:
                    cv2.imshow('img_anno', im)
                    cv2.waitKey(500)