# coding=utf-8
__author__ = 'yochin'

# memo: two objects with occlusion <- apply random background.

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

if __name__ == '__main__':
    '''
    This is the main function
    '''

    do_depth = False
    do_iaa_augmentation = False

    # DB version 7
    imageFolder = '/media/yochin/ModMan DB/ModMan DB/ETRI_HMI/Synthetic Test SceneV2(3rd yr)/generated/Duet/TestSet'
    backgFolder = '/media/yochin/ModMan DB/otherDBs/NewNegative'  # 배경: 약 8000장

    # this is for ModManDB portable storage
    saveBaseFolder = '/media/yochin/ModMan DB/ModMan DB/ETRI_HMI/Synthetic Test SceneV2(3rd yr)/generated/Duet'
    # saveBaseFolder = '/home/yochin/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/data/'
    savedFolder_Infos = os.path.join(saveBaseFolder, 'Annotations')
    savedFolder_Images = os.path.join(saveBaseFolder, 'Images')
    savedFolder_ImageSets = os.path.join(saveBaseFolder, 'ImageSets')

    if not os.path.exists(savedFolder_Infos):
        os.makedirs(savedFolder_Infos)

    if not os.path.exists(savedFolder_Images):
        os.makedirs(savedFolder_Images)

    if not os.path.exists(savedFolder_ImageSets):
        os.makedirs(savedFolder_ImageSets)

    IMAGE_SHORT_SIZE = 1200.
    NUM_DATA_PER_OBJECT = 20

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

        # for debugging
        if iBG == 100:
            break

    print('total %d were collected'%len(listBGimages))

    # k is sampled from all data.
    listFiles = list_files(imageFolder, 'png')

    numcopy = 1
    fid_sets_tr = open(savedFolder_ImageSets + '/' + 'train.txt', 'w')
    fid_sets_full = open(savedFolder_ImageSets + '/' + 'traintest.txt', 'w')
    fid_sets_ts = open(savedFolder_ImageSets + '/' + 'test.txt', 'w')
    fid_sets_miss = open(savedFolder_ImageSets + '/' + 'miss_image_list.txt', 'w')

    for iFile, strImg in enumerate(listFiles):
        num_saved_images = 0
        if num_saved_images == NUM_DATA_PER_OBJECT:
            break

        # read color data and mask info
        # strImg = Obj + '-rotate_' + '{0:04d}'.format(iImg) + '.png' # ver6
        strPathFilename = os.path.join(imageFolder, strImg)
        img = cv2.imread(strPathFilename.encode('utf-8'), cv2.IMREAD_COLOR)

        rgba = cv2.imread(strPathFilename.encode('utf-8'), cv2.IMREAD_UNCHANGED)
        # img = rgba[:, :, 0:3]
        imgMask = rgba[:,:,3]
        # cv2.imshow('mask', imgMask)
        # cv2.waitKey(10)

        # img = img[0:-1, :, :]   # to delete the bottom block line (maybe noise)

        # imgMask = img[:,:,0] | img[:,:,1] | img[:,:,2]
        trash, imgMask = cv2.threshold(imgMask, 250, 255, cv2.THRESH_BINARY)

        # cv2.imshow('img', img)
        # cv2.imshow('mask', imgMask)
        # cv2.waitKey(10)

        # if do_depth == True:
        #     # read depth data
        #     # strDepthImg = Obj + '-rotate_' + '{0:04d}'.format(iImg) + '.exr'    # ver6
        #     strDepthImg = Obj + '{0:04d}'.format(iImg) + '.exr'  # ver6
        #     strPathFilename2 = imageDepFolder + '/' + strDepthImg
        #
        #     depth = cv2.imread(strPathFilename2.encode('utf-8'), cv2.IMREAD_UNCHANGED)
        #     depth = depth[:,:,1]


        imgPoints = cv2.findNonZero(imgMask)
        min_rect = cv2.boundingRect(imgPoints)  # [x, y, w, h]
        imgCropColor_original = img[min_rect[1]:min_rect[1] + min_rect[3], min_rect[0]:min_rect[0] + min_rect[2]]
        imgCropMask_original = imgMask[min_rect[1]:min_rect[1] + min_rect[3], min_rect[0]:min_rect[0] + min_rect[2]]

        # if do_depth == True:
        #     imgCropDepth_original = depth[min_rect[1]:min_rect[1] + min_rect[3], min_rect[0]:min_rect[0] + min_rect[2]]
        # cv2.imshow("imgCrop", imgCrop)
        # cv2.waitKey(10)

        for iDup in range(numcopy):
            if do_iaa_augmentation is True:
                imgCropColor = seq.augment_images(np.expand_dims(imgCropColor_original, axis=0))
                imgCropColor = imgCropColor[0,:,:,:]
            else:
                imgCropColor = copy.copy(imgCropColor_original)

            imgCropMask = copy.copy(imgCropMask_original)
            # if do_depth == True:
            #     imgCropDepth = copy.copy(imgCropDepth_original)

            # translate, in-plane-rotate, background
            # choose random background image
            idxBG = rnd.randint(1, len(listBGimages))-1

            # init var
            find_proper_size = False
            num_try = 0

            tw = 0
            th = 0

            while True:
                # choose scale, 1.0 ~ 5.0
                scalefactor = float(rnd.randint(10, 50))*0.1
                tw = int(min_rect[2]*scalefactor)
                th = int(min_rect[3]*scalefactor)

                num_try = num_try + 1

                # check the object is not in boundary
                if tw < listBGimages[idxBG].shape[1]-128 and th < listBGimages[idxBG].shape[0]-128:
                    # refuse small object
                    if tw >= 64*8 and th >= 64*8:
                        imgCropMask_Temp = cv2.resize(imgCropMask, (tw, th))

                        # refuse the shy faces of object
                        if (float(cv2.countNonZero(imgCropMask_Temp)) / float(64*64)) > 0.5:

                            find_proper_size = True
                            break

                # to escape from infinite loop
                if num_try > 200:
                    break

            if find_proper_size == True:
                imgCropColor = cv2.resize(imgCropColor, (tw, th))
                imgCropMask = cv2.resize(imgCropMask, (tw, th), interpolation = cv2.INTER_NEAREST)
                if do_depth == True:
                    imgCropDepth = cv2.resize(imgCropDepth, (tw, th), interpolation = cv2.INTER_NEAREST)

                    # manipulate depth distance along with scalefactor
                    randOccDist = -2./3. * (scalefactor - 0.5) + 1.
                    imgCropDepth = imgCropDepth + randOccDist

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

                if do_depth == True:
                    tarImgDepth = np.ones((tarImg.shape[0], tarImg.shape[1]), dtype=np.float32) * 10
                    roi_depth = tarImgDepth[ty:ty+th, tx:tx+tw]

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

                        if do_depth == True:
                            occDepth_Warp = cv2.warpAffine(imgCropDepth, rotMat, (imgCropDepth.shape[1], imgCropDepth.shape[0]))

                            randOccDist = rnd.uniform(0.05, occDepth_Warp.min())
                            occDepth_Warp = occDepth_Warp - randOccDist    # occlusion object is in front of the target object.

                            # make depth with occlusion
                            fg1 = cv2.bitwise_and(imgCropDepth, imgCropDepth, mask=cv2.bitwise_not(occMask_Warp))
                            fg2 = cv2.bitwise_and(occDepth_Warp, occDepth_Warp, mask=occMask_Warp)
                            imgCropDepth_Occ = cv2.add(fg1, fg2)

                        if float(cv2.countNonZero(imgCropMask_Occ)) / float(cv2.countNonZero(imgCropMask)) > 0.5 and float(cv2.countNonZero(imgCropMask_Occ)) / float(cv2.countNonZero(imgCropMask)) < 0.75:
                            imgCropMask_whole = imgCropMask
                            imgCropMask = imgCropMask_Occ
                            if do_depth == True:
                                imgCropDepth = imgCropDepth_Occ
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

                if do_depth == True:
                    # augmentation in depth image
                    imgCropMask_whole_inv = cv2.bitwise_not(imgCropMask_whole)
                    img1_bg = cv2.bitwise_and(roi_depth, roi_depth, mask=imgCropMask_whole_inv)
                    img2_fg = cv2.bitwise_and(imgCropDepth, imgCropDepth, mask=imgCropMask_whole)

                    tarImgDepth[ty:ty + th, tx:tx + tw] = cv2.add(img1_bg, img2_fg)
                    tarImgDepth = tarImgDepth * 1000.        # m -> cm (x100) -> mm (x10)

                    # to visualize
                    if 0:
                        min_th = 300.
                        max_th = 1000.

                        tarImgDepth = (tarImgDepth - min_th) / (max_th-min_th) * 255
                        tarImgDepth[tarImgDepth>255] = 255
                        tarImgDepth[tarImgDepth<0] = 0
                        tarImgDepth = tarImgDepth.astype(np.uint8)
                    else:
                        tarImgDepth[tarImgDepth > 15000] = 15000
                        tarImgDepth[tarImgDepth < 0] = 0
                        tarImgDepth = tarImgDepth.astype(np.uint16)

                    # tarImgDepth = cv2.normalize(tarImgDepth, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                # cv2.imshow('tarImg', tarImg)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                strImgOnlyname = strImg

                cv2.imwrite(savedFolder_Images+'/'+ strImgOnlyname +'.jpg', tarImg)
                if do_depth == True:
                    cv2.imwrite(savedFolder_Images+'/'+ strImgOnlyname +'_depth.png', tarImgDepth)

                num_saved_images = num_saved_images + 1

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
                SubElement(tag_object, 'name').text = 'None'
                tag_bndbox = Element('bndbox')
                SubElement(tag_bndbox, 'xmin').text = str(tx)
                SubElement(tag_bndbox, 'ymin').text = str(ty)
                SubElement(tag_bndbox, 'xmax').text = str(tx+tw)
                SubElement(tag_bndbox, 'ymax').text = str(ty+th)
                tag_anno.append(tag_object)
                tag_object.append(tag_bndbox)
                ElementTree(tag_anno).write(savedFolder_Infos+'/'+strImgOnlyname+'.xml')


                fid_sets_ts.write('%s\n' % strImgOnlyname)
                fid_sets_full.write('%s\n' % strImgOnlyname)
            else:
                fid_sets_miss.write('.'.join(strImg.split('.')[0:-1]) + '_%d\n'%iDup)

    fid_sets_tr.close()
    fid_sets_ts.close()
    fid_sets_full.close()
    fid_sets_miss.close()
    # shuffle the result using nu_makeSuffleRowsText.py
