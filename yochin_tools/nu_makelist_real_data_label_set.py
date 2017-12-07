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

if __name__ == '__main__':
    '''
    This is the main function
    '''
    do_iaa_augmentation = False

    # 3rd DB
    imageFolder = '/media/yochin/ModMan DB/ModMan DB/ETRI_HMI/real_data_label_set'
    listCategory = ['strawberry', 'Papermate', 'Highland', 'Genuine', 'Mark',
                    'Expo', 'Champion', 'Orange', 'Apple', 'Cup',
                    'banana', 'Chiffon', 'Crayola', 'Scissors', 'TomatoSoup',
                    'Drill', 'Mustard', 'Waffle', 'Ace', 'airplane',
                    'Moncher', 'Cheezit', 'Chococo'
                    ]
    listCategoryKorean = ['01_딸기', '02_미라도연필', '03_하이랜드메모지', '04_스틱빨간상자', '05_어드벤쳐책',
                          '06_화이트보드지우개', '07_챔피온코퍼', '08_오렌지', '09_사과', '10_컵',
                          '11_바나나', '12_옐로시폰', '13_크레용', '14_가위', '15_토마토스프',
                          '16_드릴', '17_머스타드', '18_와플', '19_에이스', '20_비행기장난감',
                          '21_몽쉘', '22_치즈잇과자', '23_초코코']

    # list for official test
    # listCategory = ['Ace', 'Apple', 'Cheezit', 'Chiffon', 'Crayola',
    #                 'Drill', 'Genuine', 'Mustard', 'TomatoSoup', 'airplane']

    # this is for ModManDB portable storage
    # saveBaseFolder = '/media/yochin/ModMan DB/ModMan DB/ETRI_HMI/SLS.3DPose/ModMan_SLSv1/data/'
    saveBaseFolder = '/home/yochin/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/data/'
    savedFolder_Infos = saveBaseFolder + 'Annotations'
    savedFolder_Images = saveBaseFolder + 'Images'
    savedFolder_ImageSets = saveBaseFolder + 'ImageSets'

    IMAGE_SHORT_SIZE = 600.

    listTotal = []
    listTr = []
    listVl = []
    listTs = []
    for curObj in listCategoryKorean:
        curList = list_files_subdir(os.path.join(imageFolder, curObj), 'jpg')
        rnd.shuffle(curList)

        numData = len(curList)
        numTr = int(numData*0.8)
        numVl = int(numData*0.1)

        listTr.extend(curList[:numTr])
        listVl.extend(curList[numTr:numTr+numVl])
        listTs.extend(curList[numTr+numVl:])

        listTotal.extend(curList)

    fid_sets_tr = open(savedFolder_ImageSets + '/' + 'train.txt', 'w')
    fid_sets_vl = open(savedFolder_ImageSets + '/' + 'val.txt', 'w')
    fid_sets_ts = open(savedFolder_ImageSets + '/' + 'test.txt', 'w')

    for ith, ipath in enumerate(listTotal):
        if divmod(ith, 100)[1] == 0:
            print('%d/%d'%(ith, len(listTotal)))

        # read image
        img = cv2.imread(ipath, cv2.IMREAD_COLOR)

        # cv2.imshow("img", img)
        # cv2.waitKey(10)

        # do augmentation
        if do_iaa_augmentation is True:
            img = seq.augment_images(np.expand_dims(img, axis=0))
            img = img[0,:,:,:]
        else:
            # img = copy.copy(img)
            # rotate clockwise 90 degree.
            timg = cv2.transpose(img)
            img = cv2.flip(timg, 1);

        # read label text
        fid_label = open(os.path.splitext(ipath)[0] + '.txt', 'r')
        label_list = fid_label.readlines()
        fid_label.close()

        # write
        objNameKorean = ipath.split('/')[7]
        objNameEnglish = listCategoryKorean.index(objNameKorean)

        numPlace = ipath.split('/')[8].split('_')[0]


        strImgOnlyname = listCategory[objNameEnglish] + '_' + numPlace + '_' + '_'.join(ipath.split('/')[-1].split('_')[1:])
        strImgOnlyname = os.path.splitext(strImgOnlyname)[0]

        cv2.imwrite(savedFolder_Images + '/' + strImgOnlyname + '.jpg', img)

        strXmlOnlyname = strImgOnlyname + '.xml'

        tag_anno = Element('annotation')

        for iObjInfo in label_list:
            # left, top, right, bottom, classid in text file

            curObjInfo_splitted = iObjInfo.split('\r\n')[0].split('\t')
            # # original
            # tx = curObjInfo_splitted[0]
            # ty = curObjInfo_splitted[1]
            # bx = curObjInfo_splitted[2]
            # by = curObjInfo_splitted[3]
            # Obj = curObjInfo_splitted[4]

            # rotate clockwise 90 degree
            tx = img.shape[1] - int(curObjInfo_splitted[1])
            ty = int(curObjInfo_splitted[0])
            bx = img.shape[1] - int(curObjInfo_splitted[3])
            by = int(curObjInfo_splitted[2])
            Obj = curObjInfo_splitted[4]

            # check the boundary condition
            f_tx = int(np.min((tx, bx)))
            f_ty = int(np.min((ty, by)))
            f_bx = int(np.max((tx, bx)))
            f_by = int(np.max((ty, by)))

            if f_tx <= 1:
                f_tx = 2

            if f_ty <= 1:
                f_ty = 2

            if f_bx >= img.shape[1]-2:
                f_bx = img.shape[1]-3

            if f_by >= img.shape[0]-2:
                f_by = img.shape[0]-3

            # xml type writing
            # index, top, left, bottom, right, yochin: class, bb + angle 도추가되어야
            # tag_filename = Element('filename')
            tag_object = Element('object')
            SubElement(tag_object, 'name').text = Obj
            tag_bndbox = Element('bndbox')
            SubElement(tag_bndbox, 'xmin').text = str(f_tx)
            SubElement(tag_bndbox, 'ymin').text = str(f_ty)
            SubElement(tag_bndbox, 'xmax').text = str(f_bx)
            SubElement(tag_bndbox, 'ymax').text = str(f_by)
            tag_anno.append(tag_object)
            tag_object.append(tag_bndbox)

        ElementTree(tag_anno).write(savedFolder_Infos+'/'+strImgOnlyname+'.xml')

        # debug
        if len(label_list) == 0 or listCategory[objNameEnglish] != Obj:
            print(ipath)

        if ipath in listTr:
            fid_sets_tr.write('%s\n' % strImgOnlyname)
        elif ipath in listVl:
            fid_sets_vl.write('%s\n' % strImgOnlyname)
        elif ipath in listTs:
            fid_sets_ts.write('%s\n' % strImgOnlyname)
        else:
            print('some errors')

    fid_sets_tr.close()
    fid_sets_vl.close()
    fid_sets_ts.close()

    # shuffle the result using nu_makeSuffleRowsText.py
