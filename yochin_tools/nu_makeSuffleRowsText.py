import random

random.seed(42)  # get the seed from current time, if the argument is not provided.

# textfile = '/media/yochin/DataStorage/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/data/ImageSets/train.txt'
# textfile = '/home/yochin/Desktop/ModMan_ETRI/data/ImageSets/train.txt'
textfile = '/media/yochin/0d71bed3-b968-40a1-a28d-bf12275c6299/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/MultiObjectReal-110-selected/ImageSets/test.txt'

lines = open(textfile).readlines()
random.shuffle(lines)
# if you want to sort, replace random.shuffle(lines) to below
# lines = sorted(lines)
open(textfile, 'w').writelines(lines)



