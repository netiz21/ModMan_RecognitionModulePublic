# coding=utf-8
__author__ = 'yochin'

# nu_ApplyBackground.py_rgb + depth

# we used augmentation code from "https://github.com/aleju/imgaug"

import os
import sys
import cv2
import numpy as np
import copy
import random as rnd
from xml.etree.ElementTree import Element, dump, SubElement, ElementTree
import imgaug as ia
from imgaug import augmenters as iaa


# https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(image_center[:2], angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[:2],flags=cv2.INTER_LINEAR)
  return result

def list_files(path, ext):
    filelist = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            if name.endswith(ext):
                filelist.append(name)
    return filelist

def list_files_subdir(destpath, ext):
    filelist = []
    for path, subdirs, files in os.walk(destpath):
        for filename in files:
            f = os.path.join(path, filename)
            if os.path.isfile(f):
                if filename.endswith(ext):
                    filelist.append(f)
    return filelist

import codecs
import random as rnd


if __name__ == '__main__':
    '''
    This is the main function
    '''
    rnd.seed(42)
    pathBase = '/media/yochin/DataStorage/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1'
    list_fromFolders = [
                        # 'data_Synthetic_FourObjs',
                        'data_Synthetic_ThreeObjs',
                        'data_Synthetic_TwoObjs']

    list_numImages = [4575, 4575]
    list_numObjs = [3, 2]

    fid_sets_full = open(os.path.join(pathBase, 'data', 'ImageSets', 'traintest_multiObj_23.txt'), 'w')

    # read first txt list, then shuffle, and pick A
    # read second txt list, remove the previous list, then repeat the first one.
    # ...
    list_listFiles = [[] for i in range(len(list_numObjs))]

    for ith, nameDB in enumerate(list_fromFolders):
        numObjs = list_numObjs[ith]
        list_listFiles[ith] = codecs.open(os.path.join(pathBase, nameDB, 'ImageSets', 'traintest.txt')).read().split('\n')

        # delete the last one (if it is blank)
        if len(list_listFiles[ith][-1]) == 0:
            del list_listFiles[ith][-1]

        # shuffle the list
        rnd.shuffle(list_listFiles[ith])

        if ith == 0:
            list_listFiles[ith] = list_listFiles[ith][:list_numImages[ith]]
        else:
            list_filtered = []
            for ifile, filename_ext in enumerate(list_listFiles[ith]):
                filename = os.path.splitext(filename_ext)[0]
                checkFilename = '-'.join(filename.split('-')[:numObjs])
                checkPosename = '-'.join(filename.split('-')[numObjs:])

                listSameObjs = []
                for kth in range(0, ith):
                    temp = filter(lambda x: checkFilename in x, list_listFiles[0])
                    listSameObjs.extend(temp)
                listSameObjsPoses = filter(lambda x: checkPosename in x, listSameObjs)

                if len(listSameObjsPoses) == 0:
                    list_filtered.append(filename_ext)
                else:
                    print(listSameObjsPoses)
                    print(filename_ext)

            list_listFiles[ith] = list_filtered
            list_listFiles[ith] = list_listFiles[ith][:list_numImages[ith]]

    '''
    write to files
    '''
    for listfiles in list_listFiles:
        for files in listfiles:
            strImgOnlyname = os.path.splitext(files)[0]
            fid_sets_full.write('%s\n' % strImgOnlyname)

    fid_sets_full.close()

    # shuffle the result using nu_makeSuffleRowsText.py
