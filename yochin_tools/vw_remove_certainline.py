def read_list_linebyline(fname):
    with open(fname) as fid:
        content = fid.readlines()
    content = [item.rstrip('\n') for item in content]

    return content


def write_list_linebyline(fname, thelist):
    fid = open(fname, 'w')

    for item in thelist:
        fid.write('%s\n' % (item))

    fid.close()


list = read_list_linebyline('/media/yochin/DataStorage/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/train.txt')
list_new = []
for item in list:
    if 'Orange' in item:
        pass
    else:
        list_new.append(item)

write_list_linebyline('/media/yochin/DataStorage/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/train2.txt', list_new)