import sys, os
import matplotlib.pyplot as plt
import numpy as np

def moving_average(a, n=15) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

textfile = os.path.join('/home/yochin/Faster-RCNN_TF/output/_logs',
                        'faster_rcnn_end2end_VGG16_--set_EXP_DIR_VGGnet-Real-3obj_RNG_SEED_42_TRAIN.SCALES_[400,500,600,700].txt.2017-11-10_17-05-32')

lines = open(textfile).readlines()
new_lines = [row for row in lines if 'iter:' in row]

list_iter = []
list_total_loss = []
list_rpn_loss_cls = []
list_rpn_loss_box = []
list_loss_cls = []
list_loss_box = []
list_lr = []

for row in new_lines:
    parts = row.split(',')
    list_iter.append(int(parts[0].split(':')[1].split('/')[0]))
    list_total_loss.append(float(parts[1].split(':')[1]))
    list_rpn_loss_cls.append(float(parts[2].split(':')[1]))
    list_rpn_loss_box.append(float(parts[3].split(':')[1]))
    list_loss_cls.append(float(parts[4].split(':')[1]))
    list_loss_box.append(float(parts[5].split(':')[1]))
    list_lr.append(float(parts[6].split(':')[1].split('\n')[0]))


# moving average
# list_total_loss = moving_average(list_total_loss)

plt.subplot('231')
plt.plot(list_iter, list_total_loss)
plt.xlabel('iter')
plt.ylabel('total_loss')

plt.subplot('232')
plt.plot(list_iter, list_rpn_loss_cls)
plt.xlabel('iter')
plt.ylabel('list_rpn_loss_cls')

plt.subplot('233')
plt.plot(list_iter, list_rpn_loss_box)
plt.xlabel('iter')
plt.ylabel('list_rpn_loss_box')

plt.subplot('234')
plt.plot(list_iter, list_loss_cls)
plt.xlabel('iter')
plt.ylabel('list_loss_cls')

plt.subplot('235')
plt.plot(list_iter, list_loss_box)
plt.xlabel('iter')
plt.ylabel('list_loss_box')

plt.subplot('236')
plt.plot(list_iter, list_lr)
plt.xlabel('iter')
plt.ylabel('list_lr')


# plt.savefig('dummy.png')
plt.show()
plt.show()
