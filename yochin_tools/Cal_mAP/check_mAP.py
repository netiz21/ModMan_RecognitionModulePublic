import numpy as np
import matplotlib.pyplot as plt
import decimal

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


def checkDetection(infoGnd, contentQuery, detectorName):
    tp = 0
    fp = 0
    fn = 0
    for infoQuery in contentQuery:
        infoGndA = infoGnd.split()
        infoQueryA = infoQuery.split()
        boxA = list(map(int, infoGndA[1:]))
        boxB = list(map(int, infoQueryA[1:5]))
        if infoGndA[0] == detectorName and bb_intersection_over_union(boxA, boxB) > 0.5:
            tp = 1

    fp = len(contentQuery) - tp

    if infoGnd.split()[0] == detectorName:
        fn = 1 - tp

    if fp < 0 or tp < 0 or fn < 0:
        print('something wrong: fp %d, tp %d, fn %d'%(fp, tp, fn))

    return tp, fp, fn

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

# this func made main() could be called as import module.
def main():
    listObject = [
        'CheezIt',
        'Crayola',
        'Ace',
        'Chococo',
        'Genuine',
        'Drill',
        'Cup',
        'Scissors',
        'Tomatosoup',
        'Mustard'
    ]

    TH_range = range(0, 100, 3)
    # TH_range = list(frange(0, 105, decimal.Decimal('2.5')))
    confMat = np.zeros((len(TH_range), len(listObject), 3), dtype = np.int)

    k_train_range = []
    for round_ang in range(1, 25, 2):
        for height_ang in range(1, 14, 2):
            for dist in range(1, 7, 2):
                for ipr in range(1, 4, 1):
                    k = (ipr - 1) * 1872 + (dist - 1) * 312 + (height_ang - 1) * 24 + round_ang
                    k_train_range.append(k)

    k_test_range = []
    for k in range(1, 5617, 10):
        if k not in k_train_range:
            k_test_range.append(k)


    for iTH, TH in enumerate(TH_range):
        for objName in listObject:
            for k in k_test_range:
                # print('%s_%d'%(objName, k))
                strBase = 'H:/ModMan DB/ETRI_HMI/LinMod Rendering/Output/' + objName + '/' + objName + '-rotate_' + '{0:04d}'.format(k)
                strFileEst = strBase + '_est.txt'
                strFileGnd = strBase + '_gnd.txt'

                fid1 = open(strFileGnd)
                contentGnd = fid1.readlines()
                infoGnd = contentGnd[0]
                fid1.close()

                fid2 = open(strFileEst)
                contentQuery = fid2.readlines()
                fid2.close()

                for tari, tarObj in enumerate(listObject):
                    filteredQuery = []
                    for tempObj in contentQuery:
                        if tempObj.split()[0] == tarObj and float(tempObj.split()[5]) >= TH:
                            filteredQuery.append(tempObj)

                    if len(infoGnd) > 0:
                        tp, fp, fn = checkDetection(infoGnd, filteredQuery, tarObj)
                        confMat[iTH, tari, :] = confMat[iTH, tari, :] + [tp, fp, fn]

    for i, Obj in enumerate(listObject):
        print('%s prec vs recall\n'%Obj)
        y_prec = []
        x_recall = []
        for iTH, TH in enumerate(TH_range):
            # precision = tp / (tp + fp)
            # recall = tp / (tp + fn)
            precision = confMat[iTH, i, 0] / (confMat[iTH, i, 0] + confMat[iTH, i, 1])
            recall = confMat[iTH, i, 0] / (confMat[iTH, i, 0] + confMat[iTH, i, 2])
            print('%d %f %f'%(TH, precision, recall))

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


        # plt.figure(i)
        # plt.plot(x_recall, y_prec, 'r--', x_recall, y_interp_prec, 'g--')
        plt.plot(x_recall, y_interp_prec, 'g--')
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