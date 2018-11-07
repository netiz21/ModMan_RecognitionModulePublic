import random
import numpy as np

listCategory = ['Ace', 'Apple', 'Champion', 'Cheezit', 'Chiffon',
                    'Chococo', 'Crayola', 'Cup', 'Drill', 'Expo',
                    'Genuine', 'Highland', 'Mark',
                    'Moncher', 'Mustard', 'Papermate', 'Scissors',
                    'TomatoSoup', 'Waffle', 'airplane', 'banana',
                    'strawberry']

textfile = '/media/yochin/DataStorage/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/data/ImageSets/train_temp.txt'
textfile2 = '/media/yochin/DataStorage/Desktop/ModMan_DB/ETRI_HMI/ModMan_SLSv1/data/ImageSets/train_temp_new.txt'

lines = open(textfile).readlines()

new_lines = [row for row in lines if '_19_' in row]

# for place_id in range(1, 21):
#     cnt_list = []
#     for obj in listCategory:
#         cnt = 0
#         for row in lines:
#             if '_%02d_'%place_id in row and obj in row:
#                 cnt = cnt + 1
#
#         # print '%02d place, %s obj: %d'%(place_id, obj, cnt)
#         cnt_list.append(cnt)
#
#     cnt_list = np.array(cnt_list, dtype=float)
#     print( '%02d place, max:%f, min:%f, avg:%f, std:%f'%(place_id, np.max(cnt_list), np.min(cnt_list), np.mean(cnt_list), np.std(cnt_list)))


open(textfile2, 'w').writelines(new_lines)