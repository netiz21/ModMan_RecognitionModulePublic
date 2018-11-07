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

if __name__ == '__main__':
    '''
    This is the main function
    '''

    imageFolder = '/media/yochin/ModMan DB/ModMan DB/ETRI_HMI/SLS.3DPose/LinMod_Rendering'
    backgFolder = '/media/yochin/ModMan DB/otherDBs/NewNegative' # 배경: 약 8000장

    saveBaseFolder = '/home/yochin/ModMan_DB/ETRI_HMI/ModMan_SLSv1/data/'
    savedFolder_Infos = saveBaseFolder + 'Annotations'
    savedFolder_Images = saveBaseFolder + 'Images'
    savedFolder_ImageSets = saveBaseFolder + 'ImageSets'

    IMAGE_SHORT_SIZE = 600.

    listCategory = ['ace',
                    'champion',
                    'cheezit',
                    'chiffon',
                    'chococo',
                    'crayola',
                    'expo',
                    'genuine',
                    'highland',
                    'mark',
                    'moncher',
                    'papermate',
                    'waffle',
                    'cup',
                    'drill',
                    'mustard',
                    'scissors',
                    'tomatosoup']

    listBGimages = []
    listBGimagesnames = list_files(backgFolder, 'bmp')
    for iBG, strname in enumerate(listBGimagesnames):
        im = cv2.imread(backgFolder + '/' + strname)

        scalefactor = IMAGE_SHORT_SIZE/float(min(im.shape[0], im.shape[1]))
        tw = int(im.shape[1] * scalefactor)
        th = int(im.shape[0] * scalefactor)
        im = cv2.resize(im, (tw, th))

        ratio1 = float(im.shape[0])/float(im.shape[1])
        ratio2 = float(im.shape[1])/float(im.shape[0])

        if ratio1 < 2 and ratio2 < 2:
            listBGimages.append(copy.copy(im))
        else:
            print('%d / %d = %f, %f' % (im.shape[0], im.shape[1], ratio1, ratio2))

        # # for debugging
        # if iBG == 10:
        #     break

    print('total %d were collected'%len(listBGimages))

    k_train_range = []
    for round_ang in range(1, 25, 2):
        for height_ang in range(1, 14, 2):
            for dist in range(1, 7, 2):
                for ipr in range(1, 4, 1):
                    k = (ipr - 1) * 1872 + (dist - 1) * 312 + (height_ang - 1) * 24 + round_ang
                    k_train_range.append(k)

    # for im in listBGimages:
    #     cv2.imshow('im', im)
    #     cv2.waitKey(10)
    modnum = 0 # this is random number between 0 ~ 9, to select the test data among all data.
    numcopy = 1
    fid_sets_tr = open(savedFolder_ImageSets + '/' + 'train.txt', 'w')
    fid_sets_ts = open(savedFolder_ImageSets + '/' + 'test.txt', 'w')
    fid_sets_miss = open(savedFolder_ImageSets + '/' + 'miss_image_list.txt', 'w')

    for iObj, Obj in enumerate(listCategory):
        imageObjFolder = imageFolder + '/RGB/' + Obj

        for iImg in k_train_range:
            strImg = Obj + '-rotate_' + '{0:04d}'.format(iImg) + '.png'
            print(strImg)
            strPathFilename = imageObjFolder + '/' + strImg
            img = cv2.imread(strPathFilename.encode('utf-8'), cv2.IMREAD_COLOR)

            rgba = cv2.imread(strPathFilename.encode('utf-8'), cv2.IMREAD_UNCHANGED)
            # img = rgba[:, :, 0:3]
            imgMask = rgba[:,:,3]
            # cv2.imshow('mask', imgMask)
            # cv2.waitKey(10)

            # img = img[0:-1, :, :]   # to delete the bottom block line (maybe noise)

            # imgMask = img[:,:,0] | img[:,:,1] | img[:,:,2]
            trash, imgMask = cv2.threshold(imgMask, 1, 255, cv2.THRESH_BINARY)

            # cv2.imshow('img', img)
            # cv2.imshow('mask', imgMask)
            # cv2.waitKey(10)

            imgPoints = cv2.findNonZero(imgMask)
            min_rect = cv2.boundingRect(imgPoints)  # [x, y, w, h]
            imgCropColor = img[min_rect[1]:min_rect[1] + min_rect[3], min_rect[0]:min_rect[0] + min_rect[2]]
            imgCropMask = imgMask[min_rect[1]:min_rect[1] + min_rect[3], min_rect[0]:min_rect[0] + min_rect[2]]
            # cv2.imshow("imgCrop", imgCrop)
            # cv2.waitKey(10)

            for iDup in range(numcopy):
                # translate, in-plane-rotate, background
                # choose random background
                idxBG = rnd.randint(1, len(listBGimages))-1

                find_proper_size = False
                num_try = 0

                tw = 0
                th = 0

                while True:
                    # choose scale
                    scalefactor = float(rnd.randint(2, 20))*0.1
                    tw = int(min_rect[2]*scalefactor)
                    th = int(min_rect[3]*scalefactor)

                    num_try = num_try + 1

                    if tw < listBGimages[idxBG].shape[1]-128 and th < listBGimages[idxBG].shape[0]-128:
                        if tw >= 64 and th >= 64:
                            find_proper_size = True
                            break

                    if num_try > 100:
                        break


                if find_proper_size == True:
                    imgCropColor = cv2.resize(imgCropColor, (tw, th))
                    imgCropMask = cv2.resize(imgCropMask, (tw, th))

                    # choose random position
                    # print('tw:th=%d:%d'%(tw,th))
                    margin = min(tw, th)
                    # print('tw,th: %d,%d / %d,%d'%(tw,th,listBGimages[idxBG].shape[1],listBGimages[idxBG].shape[0]))
                    # print('tx:(%d,%d)'%(1, listBGimages[idxBG].shape[1]-tw))
                    # print('ty:(%d,%d)'%(1, listBGimages[idxBG].shape[0]-th))
                    tx = rnd.randint(64, listBGimages[idxBG].shape[1]-tw-64)-1    # zero-based
                    ty = rnd.randint(64, listBGimages[idxBG].shape[0]-th-64)-1    # zero-based

                    # background합성
                    # set ROI
                    tarImg = copy.copy(listBGimages[idxBG])
                    roi = tarImg[ty:ty+th, tx:tx+tw]

                    # mask and inv_mask
                    imgCropMask_inv = cv2.bitwise_not(imgCropMask)

                    # delete black area
                    img1_bg = cv2.bitwise_and(roi,roi,mask=imgCropMask_inv)
                    # cv2.imshow('roi', roi)
                    # cv2.imshow('Mask_inv', imgCropMask_inv)

                    img2_fg = cv2.bitwise_and(imgCropColor, imgCropColor, mask=imgCropMask)

                    dst = cv2.add(img1_bg, img2_fg)
                    tarImg[ty:ty+th, tx:tx+tw] = dst

                    # cv2.imshow('tarImg', tarImg)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    strImgOnlyname = Obj + '-rotate_' + '{0:04d}'.format(iImg) + '_%d'%iDup

                    cv2.imwrite(savedFolder_Images+'/'+ strImgOnlyname +'.jpg', tarImg)

                    # text type writing
                    # f = open(savedFolder_Infos+'/'+'.'.join(strImg.split('.')[0:-1])+'.txt', 'w')
                    # # index, top, left, bottom, right, yochin: class, bb + angle 도추가되어야
                    # f.write('%s %d %d %d %d'%(Obj, ty, tx, ty+th, tx+tw))
                    # f.close()

                    # xml type writing
                    # index, top, left, bottom, right, yochin: class, bb + angle 도추가되어야
                    tag_anno = Element('annotation')
                    # tag_filename = Element('filename')
                    tag_object = Element('object')
                    SubElement(tag_object, 'name').text = Obj
                    tag_bndbox = Element('bndbox')
                    SubElement(tag_bndbox, 'xmin').text = str(tx)
                    SubElement(tag_bndbox, 'ymin').text = str(ty)
                    SubElement(tag_bndbox, 'xmax').text = str(tx+tw)
                    SubElement(tag_bndbox, 'ymax').text = str(ty+th)
                    tag_anno.append(tag_object)
                    tag_object.append(tag_bndbox)
                    ElementTree(tag_anno).write(savedFolder_Infos+'/'+strImgOnlyname+'.xml')

                    if(((iImg % 10) == 0) and (iDup == 0)):
                        modnum = rnd.randint(0, 9)

                    if((iImg % 10) == modnum):
                        fid_sets_ts.write('%s\n'%strImgOnlyname)
                    else:
                        fid_sets_tr.write('%s\n'%strImgOnlyname)
                else:
                    fid_sets_miss.write('.'.join(strImg.split('.')[0:-1]) + '_%d\n'%iDup)

    fid_sets_tr.close()
    fid_sets_ts.close()
    fid_sets_miss.close()
    # shuffle the result using nu_makeSuffleRowsText.py
