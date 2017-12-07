import random

random.seed(42)  # get the seed from current time, if the argument is not provided.

textfile = '/media/yochin/DataStorage/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/data/ImageSets/train.txt'

lines = open(textfile).readlines()
random.shuffle(lines)
# if you want to sort, replace random.shuffle(lines) to below
# lines = sorted(lines)
open(textfile, 'w').writelines(lines)



