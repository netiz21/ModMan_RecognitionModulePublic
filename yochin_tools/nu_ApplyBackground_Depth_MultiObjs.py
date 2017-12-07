# coding=utf-8
__author__ = 'yochin'

# memo: multiple objects with occlusion <- apply random background.

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

# random example images
images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        # iaa.Fliplr(0.5), # horizontally flip 50% of all images
        # iaa.Flipud(0.2), # vertically flip 20% of all images
        # sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
        # sometimes(iaa.Affine(
        #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
        #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
        #     rotate=(-45, 45), # rotate by -45 to +45 degrees
        #     shear=(-16, 16), # shear by -16 to +16 degrees
        #     order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
        #     cval=(0, 255), # if mode is constant, use a cval between 0 and 255
        #     mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        # )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges
                # sometimes(iaa.OneOf([
                #     iaa.EdgeDetect(alpha=(0, 0.7)),
                #     iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                # ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                # iaa.OneOf([
                #     iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                #     iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                # ]),
                iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                iaa.Grayscale(alpha=(0.0, 1.0))
                # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
            ],
            random_order=True
        )
    ],
    random_order=True
)

def list_files(path, ext):
    filelist = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            if name.endswith(ext):
                filelist.append(name)
    return filelist

def getBoundingBoxfromAlphaImage(imgMask):
    trash, imgMask = cv2.threshold(imgMask, 250, 255, cv2.THRESH_BINARY)
    imgPoints = cv2.findNonZero(imgMask)
    min_rect = cv2.boundingRect(imgPoints)  # [x, y, w, h]

    return min_rect


if __name__ == '__main__':
    '''
    This is the main function
    '''
    # 4 object is subset of 3 object
    # 3 object is subset of 2 object
    # First, divide the number of images for each set (4 object 33%, 3 object 33%, 2 object 33%)
    # Second, make lists which are images for each set and do not share ths images.
    # Third, make image sets
    rnd.seed(42)  # get the seed from current time, if the argument is not provided.

    '''
    Settings
    '''
    do_iaa_augmentation = False

    nNumMultiple = 4
    pathImageFolders = '/media/yochin/ModMan DB/ModMan DB/ETRI_HMI/SLS.3DPose/SLSv1.MultiObjectsSet_%d'%nNumMultiple
    backgFolder = '/media/yochin/ModMan DB/otherDBs/NewNegative'  # 배경: 약 8000장
    listCategory = ['Ace', 'Apple', 'Champion', 'Cheezit', 'Chiffon',
                    'Chococo', 'Crayola', 'Cup', 'Drill', 'Expo',
                    'Genuine', 'Highland', 'Mark',
                    'Moncher', 'Mustard', 'Papermate', 'Scissors',
                    'TomatoSoup', 'Waffle', 'airplane', 'banana',
                    'strawberry']

    # this is for ModManDB portable storage
    saveBaseFolder = '/media/yochin/DataStorage/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/data/'
    savedFolder_Infos = os.path.join(saveBaseFolder, 'Annotations')
    savedFolder_Images = os.path.join(saveBaseFolder, 'Images')
    savedFolder_ImageSets = os.path.join(saveBaseFolder, 'ImageSets')

    if not os.path.exists(savedFolder_Infos):
        os.makedirs(savedFolder_Infos)

    if not os.path.exists(savedFolder_Images):
        os.makedirs(savedFolder_Images)

    if not os.path.exists(savedFolder_ImageSets):
        os.makedirs(savedFolder_ImageSets)

    IMAGE_SHORT_SIZE = 600.

    numcopy = 1

    '''
    load background image
    '''
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

    print('total background images %d were collected'%len(listBGimages))

    # k is sampled from all data.
    listFiles = list_files(os.path.join(pathImageFolders, 'Images'), 'png')

    fid_sets_full = open(savedFolder_ImageSets + '/' + 'traintest.txt', 'w')
    fid_sets_miss = open(savedFolder_ImageSets + '/' + 'miss_image_list.txt', 'w')

    for strImg in listFiles:
        # read color data and mask info
        print(strImg)
        strPathFilename = os.path.join(pathImageFolders, 'Images') + '/' + strImg
        img = cv2.imread(strPathFilename.encode('utf-8'), cv2.IMREAD_COLOR)

        # Masks
        rgba = cv2.imread(strPathFilename.encode('utf-8'), cv2.IMREAD_UNCHANGED)
        imgMask = rgba[:, :, 3]
        min_rect = getBoundingBoxfromAlphaImage(imgMask)

        imgCropColor_original = img[min_rect[1]:min_rect[1] + min_rect[3],
                                min_rect[0]:min_rect[0] + min_rect[2]]
        imgCropMask_original = imgMask[min_rect[1]:min_rect[1] + min_rect[3],
                               min_rect[0]:min_rect[0] + min_rect[2]]

        for iDup in range(numcopy):
            if do_iaa_augmentation is True:
                imgCropColor = seq.augment_images(np.expand_dims(imgCropColor_original, axis=0))
                imgCropColor = imgCropColor[0,:,:,:]
            else:
                imgCropColor = copy.copy(imgCropColor_original)

            imgCropMask = copy.copy(imgCropMask_original)

            # translate, in-plane-rotate, background
            # choose random background image
            idxBG = rnd.randint(1, len(listBGimages))-1

            # init var
            find_proper_size = True
            num_try = 0

            tw = int(min_rect[2])
            th = int(min_rect[3])

            if find_proper_size == True:
                # choose random position
                # print('tw:th=%d:%d'%(tw,th))
                margin = min(tw, th)
                # print('tw,th: %d,%d / %d,%d'%(tw,th,listBGimages[idxBG].shape[1],listBGimages[idxBG].shape[0]))
                # print('tx:(%d,%d)'%(1, listBGimages[idxBG].shape[1]-tw))
                # print('ty:(%d,%d)'%(1, listBGimages[idxBG].shape[0]-th))
                tx = int(min_rect[0])    # zero-based
                ty = int(min_rect[1])    # zero-based

                # background합성
                # set ROI
                tarImg = copy.copy(listBGimages[idxBG])
                roi = tarImg[ty:ty+th, tx:tx+tw]

                # with 80% probability, hide mask with random rotation, scaling.
                if 0:#rnd.randint(0, 9) > 4.5:
                    while True:
                        cx = rnd.randint(1, tw-2)
                        cy = rnd.randint(1, th-2)
                        rndangle = rnd.randint(0, 180)
                        rndscale = float(rnd.randint(15, 40)) / 10.
                        rotMat = cv2.getRotationMatrix2D((cx, cy), rndangle, rndscale)

                        # occMask: target 255, occlusion 0,
                        occMask_Warp = cv2.warpAffine(imgCropMask, rotMat, (imgCropMask.shape[1], imgCropMask.shape[0]))
                        _, occMask_Warp = cv2.threshold(occMask_Warp, 250, 255, cv2.THRESH_BINARY)
                        occMask = cv2.bitwise_not(occMask_Warp)
                        imgCropMask_Occ = cv2.bitwise_and(imgCropMask, occMask)
                        _, imgCropMask_Occ = cv2.threshold(imgCropMask_Occ, 250, 255, cv2.THRESH_BINARY)

                        if float(cv2.countNonZero(imgCropMask_Occ)) / float(cv2.countNonZero(imgCropMask)) > 0.5 and float(cv2.countNonZero(imgCropMask_Occ)) / float(cv2.countNonZero(imgCropMask)) < 0.75:
                            imgCropMask_whole = imgCropMask
                            imgCropMask = imgCropMask_Occ
                            break
                else:
                    imgCropMask_whole = imgCropMask

                # clear edge
                _, imgCropMask = cv2.threshold(imgCropMask, 250, 255, cv2.THRESH_BINARY)

                # delete black area and final augmentation
                imgCropMask_inv = cv2.bitwise_not(imgCropMask)
                img1_bg = cv2.bitwise_and(roi,roi,mask=imgCropMask_inv)
                img2_fg = cv2.bitwise_and(imgCropColor, imgCropColor, mask=imgCropMask)

                dst = cv2.add(img1_bg, img2_fg)
                tarImg[ty:ty+th, tx:tx+tw] = dst

                # cv2.imshow('tarImg', tarImg)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                strImgOnlyname = strImg

                cv2.imwrite(savedFolder_Images+'/'+ strImgOnlyname +'.jpg', tarImg)

                # text type writing
                # f = open(savedFolder_Infos+'/'+'.'.join(strImg.split('.')[0:-1])+'.txt', 'w')
                # # index, top, left, bottom, right, yochin: class, bb + angle 도추가되어야
                # f.write('%s %d %d %d %d'%(Obj, ty, tx, ty+th, tx+tw))
                # f.close()

                # xml type writing
                # index, top, left, bottom, right, yochin: class, bb + angle 도추가되어야
                tag_anno = Element('annotation')
                for iAnno in range(1, nNumMultiple+1):
                    strPathFilenameMask1 = os.path.join(pathImageFolders, 'Masks') + '/' + os.path.splitext(strImg)[0] + '-%d.png'%iAnno
                    rgba = cv2.imread(strPathFilenameMask1.encode('utf-8'), cv2.IMREAD_UNCHANGED)
                    imgMask = rgba[:, :, 3]
                    min_rect_obj1 = getBoundingBoxfromAlphaImage(imgMask)

                    # tag_filename = Element('filename')
                    tag_object = Element('object')
                    SubElement(tag_object, 'name').text = os.path.splitext(strImg)[0].split('-')[iAnno-1]
                    tag_bndbox = Element('bndbox')

                    tx = min_rect_obj1[0]
                    ty = min_rect_obj1[1]
                    tw = min_rect_obj1[2]
                    th = min_rect_obj1[3]

                    SubElement(tag_bndbox, 'xmin').text = str(tx)
                    SubElement(tag_bndbox, 'ymin').text = str(ty)
                    SubElement(tag_bndbox, 'xmax').text = str(tx+tw)
                    SubElement(tag_bndbox, 'ymax').text = str(ty+th)
                    tag_anno.append(tag_object)
                    tag_object.append(tag_bndbox)

                ElementTree(tag_anno).write(savedFolder_Infos+'/'+strImgOnlyname+'.xml')

                fid_sets_full.write('%s\n' % strImgOnlyname)
            else:
                fid_sets_miss.write('.'.join(strImg.split('.')[0:-1]) + '_%d\n'%iDup)

    fid_sets_full.close()
    fid_sets_miss.close()
    # shuffle the result using nu_makeSuffleRowsText.py
