import random

textfile = '/home/yochin/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/data/ImageSets/train.txt'

lines = open(textfile).readlines()
random.shuffle(lines)
open(textfile, 'w').writelines(lines)
