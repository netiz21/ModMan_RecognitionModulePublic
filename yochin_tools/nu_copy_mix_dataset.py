import os
import shutil

basePath = '/media/yochin/DataStorage/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1'

list_subDir = ['Annotations', 'ImageSets', 'Images']

dstPath = os.path.join(basePath, 'data')
srcPath = os.path.join(basePath, 'data_SynthSingleObjectVarEnvVer2')

# make directory in dstPath
if not os.path.exists(dstPath):
    os.makedirs(dstPath)

    for subDir in list_subDir:
        os.makedirs(os.path.join(dstPath, subDir))

# copy from srcPath *** WILL OVERWRITE ***
for subDir in list_subDir:
    print(subDir)
    src_files = os.listdir(os.path.join(srcPath, subDir))
    for filename in src_files:
        path_filename = os.path.join(srcPath, subDir, filename)

        if os.path.isfile(path_filename) is True:
            shutil.copy(path_filename, os.path.join(dstPath, subDir))