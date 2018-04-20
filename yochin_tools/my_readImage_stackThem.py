import cv2, os
import numpy as np

def list_files(path, ext):
    filelist = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            if name.endswith(ext):
                filelist.append(name)
    return filelist


strImageFolder = '/media/yochin/ModMan DB2/ModMan DB/UW RGBD/rgbd-dataset_eval/apple/apple_1'

listTestFiles = list_files(strImageFolder, 'png')

trainX = np.empty((0, 3, 24, 24), float)
trainY = np.empty((0, 1), int)

for filenameExt in listTestFiles:
    im = cv2.imread(os.path.join(strImageFolder, filenameExt))	# H x W x C
    # im = scipy.ndimage.imread(strImageFolder, mode='RGB')
    im = cv2.resize(im, (24, 24))
    # im = scipy.misc.imresize(im, (24, 24))

    label = 1

    im = im.transpose((2,0,1))

    trainX = np.append(trainX, im, axis=0)
    trainY = np.append(trainY, label, axis=0)