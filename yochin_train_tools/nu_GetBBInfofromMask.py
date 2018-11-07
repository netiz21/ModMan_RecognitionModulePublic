# coding=utf-8
__author__ = 'yochin'

# 물체만존재하는영상에서위치와배경을조절한영상으로수정

import os
import sys
import cv2
import copy
import random as rnd
from xml.etree.ElementTree import Element, dump, SubElement, ElementTree

def list_files(path, ext):
    filelist = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            if name.endswith(ext):
                filelist.append(name)
    return filelist

def list_files_subdir(destpath, ext, partialFilename):
    filelist = []
    for path, subdirs, files in os.walk(destpath):
        for filename in files:
            f = os.path.join(path, filename)
            if os.path.isfile(f):
                if filename.endswith(ext):
                    if partialFilename in filename:
                        filelist.append(f)

    return filelist

def find_exact_name(list, query):
    for obj in list:
        if obj.lower() == query.lower():
            return obj

    return ''

if __name__ == '__main__':
    '''
    This is the main function
    '''

    imageFolder = '/home/yochin/Desktop/data_synth10/Render 007 10-Scene 9-Env 20 Frame'

    saveBaseFolder = '/home/yochin/Desktop/data_synth10'
    savedFolder_Infos = os.path.join(saveBaseFolder, 'Annotations')
    savedFolder_Images = os.path.join(saveBaseFolder, 'Images')
    savedFolder_ImageSets = os.path.join(saveBaseFolder, 'ImageSets')

    IMAGE_SHORT_SIZE = 600.

    listCategory = ['Ace',
                    'Apple',
                    'Cheezit',
                    'Chiffon',
                    'Crayola',
                    'Genuine',
                    'Drill',
                    'Mustard',
                    'TomatoSoup',
                    'airplane']
    listCategoryLower = [item.lower() for item in listCategory]

    if not os.path.exists(savedFolder_Infos):
        os.makedirs(savedFolder_Infos)

    if not os.path.exists(savedFolder_Images):
        os.makedirs(savedFolder_Images)

    if not os.path.exists(savedFolder_ImageSets):
        os.makedirs(savedFolder_ImageSets)

    fid_sets_tr = open(savedFolder_ImageSets + '/' + 'train.txt', 'w')
    fid_sets_ts = open(savedFolder_ImageSets + '/' + 'test.txt', 'w')
    fid_sets_miss = open(savedFolder_ImageSets + '/' + 'miss_image_list.txt', 'w')

    ll = list_files_subdir(imageFolder, 'png', 'RGB')

    for pathfile in ll:
        path = '/'.join(pathfile.split('/')[:-2])
        filename = pathfile.split('/')[-1]
        filenameOnly = filename.split('.')[0]

        lightname = pathfile.split('/')[-3]
        scenename = pathfile.split('/')[-6]

        img = cv2.imread(pathfile.encode('utf-8'), cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(savedFolder_Images, '%s_%s_%s' % (scenename, lightname, filename)), img)

        filename_id = filename.split('.')[0].split('_')[-1]

        path_rep = os.path.join('/'.join(path.split('/')[:-1]), 'bathroom_2k.hdr', 'Mask')

        list_mask = list_files(path_rep, 'png')

        # xml type writing
        # index, top, left, bottom, right, yochin: class, bb + angle 도추가되어야
        tag_anno = Element('annotation')
        # tag_filename = Element('filename')

        for maskname in list_mask:
            _, mask_obj, mask_id = maskname.split('.')[0].split('_')       # _, Ace, 0101

            if filename_id == mask_id:
                if mask_obj.lower() in listCategoryLower:
                    imgMask = cv2.imread(os.path.join(path_rep, maskname).encode('utf-8'), cv2.IMREAD_UNCHANGED)

                    # cv2.imshow('mask', imgMask)
                    # cv2.waitKey(10)

                    trash, imgMask = cv2.threshold(imgMask, 1, 255, cv2.THRESH_BINARY)

                    imgPoints = cv2.findNonZero(imgMask)

                    if imgPoints is not None and len(imgPoints) > 200:    # 200 = 20 x 10
                        min_rect = cv2.boundingRect(imgPoints)  # [x, y, w, h]

                        # # xml type writing
                        # # index, top, left, bottom, right, yochin: class, bb + angle 도추가되어야
                        # tag_anno = Element('annotation')
                        # # tag_filename = Element('filename')
                        tag_object = Element('object')
                        SubElement(tag_object, 'name').text = find_exact_name(listCategory, mask_obj)
                        tag_bndbox = Element('bndbox')
                        SubElement(tag_bndbox, 'xmin').text = str(min_rect[0])
                        SubElement(tag_bndbox, 'ymin').text = str(min_rect[1])
                        SubElement(tag_bndbox, 'xmax').text = str(min_rect[0]+min_rect[2])
                        SubElement(tag_bndbox, 'ymax').text = str(min_rect[1]+min_rect[3])
                        tag_anno.append(tag_object)
                        tag_object.append(tag_bndbox)

                        fid_sets_tr.write('%s\n'%'%s_%s_%s'%(scenename, lightname, filenameOnly))
                    else:
                        fid_sets_miss.write('%s\n'%os.path.join(path_rep, maskname))

        ElementTree(tag_anno).write(savedFolder_Infos + '/' + '%s_%s_%s'%(scenename, lightname, filenameOnly) + '.xml')

    fid_sets_tr.close()
    fid_sets_ts.close()
    fid_sets_miss.close()
    # shuffle the result using nu_makeSuffleRowsText.py
