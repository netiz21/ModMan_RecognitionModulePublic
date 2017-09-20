# coding=utf-8
# #!/usr/bin/env python

import os
from xml.etree.ElementTree import Element, dump, SubElement, ElementTree

def list_files(path, ext):
    filelist = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            if name.endswith(ext):
                filelist.append(name)
    return filelist


# read all txt file
strPathRoot = '/media/yochin/ModMan DB/ModMan DB/ETRI_HMI/real_data_label_set_Total/test_set'

listTxt = list_files(strPathRoot, 'txt')

# save xml file: nu_background
for gndTxt in listTxt:
    listBBs = open(os.path.join(strPathRoot, gndTxt), 'r').read().split('\n')
    gndXml = os.path.splitext(gndTxt)[0] + '.xml'

    if len(listBBs[-1]) == 0:
        del listBBs[-1]

    tag_anno = Element('annotation')

    for curBB in listBBs:
        curBB = curBB.split('\t')
    # writing part in C++
    # for (size_t j = 0; j < boxes.size(); ++j) {
    # fout << (int)((float)boxes[j].tl().x / g_scale)
    # << '\t' << (int)((float)boxes[j].tl().y / g_scale)
    # << '\t' << (int)((float)boxes[j].br().x / g_scale)
    # << '\t' << (int)((float)boxes[j].br().y / g_scale)
    # << '\t' << class_per_box[j]
    # << endl;
    # }

    # xml type writing
    # index, top, left, bottom, right, yochin: class, bb + angle 도추가되어야

        # tag_filename = Element('filename')
        tag_object = Element('object')
        SubElement(tag_object, 'name').text = curBB[4]
        tag_bndbox = Element('bndbox')
        SubElement(tag_bndbox, 'xmin').text = curBB[0]
        SubElement(tag_bndbox, 'ymin').text = curBB[1]
        SubElement(tag_bndbox, 'xmax').text = curBB[2]
        SubElement(tag_bndbox, 'ymax').text = curBB[3]
        tag_anno.append(tag_object)
        tag_object.append(tag_bndbox)

    ElementTree(tag_anno).write(os.path.join(strPathRoot, gndXml))
#