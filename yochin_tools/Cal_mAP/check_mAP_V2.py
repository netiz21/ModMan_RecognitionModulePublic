import numpy as np
import matplotlib.pyplot as plt
import decimal
import os
from xml.etree.ElementTree import Element, dump, parse

# http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(rectA, rectB):
    boxA = [rectA[0], rectA[1], rectA[0]+rectA[2], rectA[1]+rectA[3]]
    boxB = [rectB[0], rectB[1], rectB[0]+rectB[2], rectB[1]+rectB[3]]
    # boxA[0,1] = left-top point, x, y
    # boxB[0,1] = right-bottom point, x, y
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# http://stats.stackexchange.com/questions/260430/average-precision-in-object-detection
# https://sanchom.wordpress.com/tag/average-precision/
# iterate through entire list of predictions for all images
#
# if IOU > threshold
#     if object not detected yet
#         TP++
#     else
#         FP++    // nms should reduce # of overlapping predictions
# else
#     FP++
#
# if no prediction made for an image
#     FN++
#
# Precision = TP/(TP+FP)
# Recall = TP/(TP+FN)

# check name and IoU is accepted.
def checkDetection(infoGnd, infoQuery):
    infoGndA = infoGnd.split()
    infoQueryA = infoQuery.split()

    boxA = list(map(int, infoGndA[1:]))
    boxB = list(map(int, infoQueryA[1:5]))

    ret = False

    if infoGndA[0].lower() == infoQueryA[0].lower() and bb_intersection_over_union(boxA, boxB) > 0.5:
        ret = True

    return ret

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

def list_files(path, ext):
    filelist = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            if name.endswith(ext):
                filelist.append(name)
    return filelist

'''
contentQuery = list of [Objectname, lt.x, lt.y, rb.x, rb.y]
'''
def read_data_txt(strfilename):
    fid2 = open(strfilename)
    contentQuery = fid2.readlines()
    fid2.close()

    return contentQuery


def read_data_xml(strfilename):
    tree = parse(strfilename)
    root = tree.getroot()

    retlist = []

    for obj in root.findall('object'):

        objname = obj.findtext('name')
        bndbox = obj.find('bndbox')
        xmin = bndbox.findtext('xmin')
        ymin = bndbox.findtext('ymin')
        xmax = bndbox.findtext('xmax')
        ymax = bndbox.findtext('ymax')

        score = obj.findtext('score')

        if score:
            info = '%s %s %s %s %s %s'%(objname, xmin, ymin, xmax, ymax, score)
        else:
            info = '%s %s %s %s %s' % (objname, xmin, ymin, xmax, ymax)
        retlist.append(info)

    return retlist


# this func made main() could be called as import module.
# read all files in Images and check both estDir and gndDir and calculate mAP
def main():
    listObject = [
           'strawberry', 'papermate', 'highland', 'genuine', 'mark',
           'expo', 'champion', 'orange', 'apple', 'cup',
           'banana', 'chiffon', 'crayola', 'scissors', 'tomatosoup',
           'drill', 'mustard', 'waffle', 'ace', 'airplane',
           'moncher', 'cheezit', 'chococo', 'sponge'
    ]

    TH_range = np.arange(0., 1.025, 0.025)
    confMat = np.zeros((len(TH_range), len(listObject), 3), dtype = np.int)

    strImageFolder = '/media/yochin/ModMan DB/ModMan DB/ETRI_HMI/Synthetic Test SceneV2(3rd yr)/TestSet/Total200/Images/'       # read this image files and make a file list
    strEstFolder = '/media/yochin/ModMan DB/ModMan DB/ETRI_HMI/Synthetic Test SceneV2(3rd yr)/TestSet/Total200/estResult/'      # check this your answer
    strGndFolder = '/media/yochin/ModMan DB/ModMan DB/ETRI_HMI/Synthetic Test SceneV2(3rd yr)/TestSet/Total200/Annotations/'    # compare with this groundtruth
    listTestFiles = list_files(strImageFolder, 'jpg')

    for iTH, TH in enumerate(TH_range):
        TH = TH * 0.01
        for filename in listTestFiles:
            # print('%s_%d'%(objName, k))
            strBase = os.path.splitext(filename)[0]
            strFileEst = strEstFolder + strBase + '_est.xml'
            strFileGnd = strGndFolder + strBase + '.xml'

            # fid1 = open(strFileGnd)
            # contentGnd = fid1.readlines()
            # infoGnd = contentGnd[0]
            # fid1.close()

            # gnd
            contentGnd = read_data_xml(strFileGnd)

            # est
            contentQuery = read_data_xml(strFileEst)

            # for each candidate object
            filteredQuery = []
            for tari, tarObj in enumerate(listObject):
                # filter in the object in the candidate and having higher score.
                for tempObj in contentQuery:
                    if tempObj.split()[0].lower() == tarObj.lower() and float(tempObj.split()[5]) >= TH:
                        filteredQuery.append(tempObj)

                # lists: contentGnd, filteredQuery

            # first all objected are detected
            for iGnd, infoGnd in enumerate(contentGnd):
                for iQuery, infoQuery in enumerate(filteredQuery):
                    if len(infoGnd) > 0 and len(infoQuery) > 0:
                        if checkDetection(infoGnd, infoQuery) is True:
                            # increase tp
                            tarname = infoGnd.split()[0].lower()
                            idx = listObject.index(tarname)

                            # 0: tp, 1:fp, 2:fn
                            confMat[iTH, idx, 0] = confMat[iTH, idx, 0] + 1

                            # delete in filteredQuery, contentGnd
                            contentGnd[iGnd] = ''
                            filteredQuery[iQuery] = ''


            # second check remainders and increase false positive and negative
            for iGnd, infoGnd in enumerate(contentGnd):
                if len(infoGnd) > 0:
                    # increase fn
                    tarname = infoGnd.split()[0].lower()
                    idx = listObject.index(tarname)
                    # 0: tp, 1:fp, 2:fn
                    confMat[iTH, idx, 2] = confMat[iTH, idx, 2] + 1

            for iQuery, infoQuery in enumerate(filteredQuery):
                if len(infoQuery) > 0:
                    # increase fp
                    tarname = infoQuery.split()[0].lower()
                    idx = listObject.index(tarname)
                    # 0: tp, 1:fp, 2:fn
                    confMat[iTH, idx, 1] = confMat[iTH, idx, 1] + 1




    for i, Obj in enumerate(listObject):
        print('%s prec vs recall\n'%Obj)
        y_prec = []
        x_recall = []
        for iTH, TH in enumerate(TH_range):
            # precision = tp / (tp + fp)
            # recall = tp / (tp + fn)
            precision = float(confMat[iTH, i, 0]) / float(confMat[iTH, i, 0] + confMat[iTH, i, 1])
            recall = float(confMat[iTH, i, 0]) / float(confMat[iTH, i, 0] + confMat[iTH, i, 2])
            print('%f %f %f'%(TH, precision, recall))

            y_prec.append(precision)
            x_recall.append(recall)

        y_interp_prec = []
        for iTH in range(0, len(y_prec)):
            y_interp_prec.append(max(y_prec[0:iTH+1]))

        # set start and end point condition
        x_recall.append(0.0)
        y_interp_prec.append(max(y_interp_prec))

        x_recall.insert(0, 1.0)
        y_interp_prec.insert(0, 0.0)

        # calculate AP
        listPrec = []
        for recTH in range(0, 11):
            listPrec.append(max([y_interp_prec[index] for index, value in enumerate(x_recall) if recTH * 0.1 <= value]))

        AP = sum(listPrec)/float(len(listPrec))
        print('AP: %f'%AP)
        print(confMat[iTH, i, :])
        print(sum(confMat[iTH, i, :]))


        # plt.figure(i)
        # plt.plot(x_recall, y_prec, 'r--', x_recall, y_interp_prec, 'g--')
        plt.plot(x_recall, y_interp_prec, 'g*-', linewidth=4.0)
        plt.axis([0, 1, 0, 1])
        plt.xlabel('recall')
        plt.ylabel('precision')
        # plt.legend('precision', 'interp_precision')
        plt.title(Obj)
        plt.savefig('%s_result.png' % Obj)
        plt.show()



# this sentence made that main is not called when it is imported
if __name__ == '__main__':
    main()