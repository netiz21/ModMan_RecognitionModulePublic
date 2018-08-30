import os
import datetime

# delete t.he previous data caches
# os.system('rm ./data/cache/*.pkl')

'''
This script is for Synthetic Learning dataset
'''
# # print the start time on the file
# fid = open('/home/yochin/Desktop/ModMan_KIRIA/log2.txt', 'w')
# print >> fid, 'start time : ', datetime.datetime.now()
#
# os.system('python ./yochin_tools/nu_ApplyBackground_Depth.py')
# os.system('python ./yochin_tools/nu_ApplyBackground_Depth_KIRIA.py')
# os.system('python ./yochin_tools/nu_ApplyBackground_VarEnv.py')

# os.system('python ./yochin_tools/nu_makeSuffleRowsText.py')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.system('CUDA_VISIBLE_DEVICES=0 ./experiments/scripts/faster_rcnn_end2end.sh gpu 0 VGG16 slsv1 --set EXP_DIR VGGnet-Test RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"')


# os.system('./experiments/scripts/faster_rcnn_end2end.sh gpu 0 VGG16 slsv1 --set EXP_DIR VGGnet_KIRIA_noFlipped_DBV10_train RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"')
#  2>&1 | tee ./log_VarEnv_withTwoObjs.log

# # print the end time on the file
# print >> fid, '', datetime.datetime.now()
# fid.close()

# tensorboard, localhost:6006
# os.system('tensorboard --logdir=./logs/')

# # # Demo code by author
# os.system('python ./faster_rcnn/demo.py \
#             --net Resnet50_test \
#             --model /home/yochin/Desktop/ModManAPP_TF/output/Resnet_scriptItself/voc_2007_trainval/Resnet50_iter_140000.ckpt')
#             # --model ./data/pretrain_model/Resnet_iter_200000.ckpt')
#
# pathDB = '/media/yochin/DataStorage/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/data/ImageSets'
# os.rename('%s/train.txt'%pathDB, '%s/train_RealSingle_SynthMultiObj234.txt'%pathDB)
# os.rename('%s/train_RealSingle_SynthMultiObj23.txt'%pathDB, '%s/train.txt'%pathDB)
#
# os.system('rm ./data/cache/*.pkl')
# # os.system('python ./yochin_tools/nu_makeSuffleRowsText.py')
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.system('CUDA_VISIBLE_DEVICES=1 ./experiments/scripts/faster_rcnn_end2end.sh gpu 1 VGG16 slsv1 --set EXP_DIR VGGnet-RealSingle_SynthMultiObj23 RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"')
