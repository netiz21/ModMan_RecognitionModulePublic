PATH_BASE = '/home/jang/smallcorgi_Faster-RCNN_TF_yochin'
#PATH_BASE = '/home/yochin/Faster-RCNN_TF'
# PATH_BASE = '/home/etrimodman/smallcorgi_Faster-RCNN_TF_yochin'

'''
Objecct detection
'''
# db for real & synth data - 22+1 OBJECTS exclude 'sponge', 'orange'
CLASSES = ('__background__', # always index 0
           'strawberry', 'papermate', 'highland', 'genuine', 'mark',
           'expo', 'champion', 'apple', 'cup',
           'banana', 'chiffon', 'crayola', 'scissors', 'tomatosoup',
           'drill', 'mustard', 'waffle', 'ace', 'airplane',
           'moncher', 'cheezit', 'chococo') # 'sponge'

Candidate_CLASSES = (
    'cheezit', 'mustard', 'tomatosoup',
    'ace', 'cheezit', 'crayola', 'drill', 'moncher', 'mustard', 'waffle', 'chiffon', 'genuine', 'tomatosoup'
)

# pose estimation v3 dataset
# 'cheezit', 'mustard', 'tomatosoup'
# pose estimation v1 dataset
# above + 'ace', 'cheezit', 'crayola', 'drill', 'moncher', 'mustard', 'waffle', 'chiffon', 'genuine', 'tomatosoup',


    # 'airplane'
    #'strawberry', 'papermate', 'highland', 'mark',
           # 'expo', 'champion', 'apple', 'cup',
           # 'banana', 'scissors',
           # 'moncher', 'chococo')

# gripping points are stored.
# [y, x, z] is a mm centered at the center of the object
ListGrippingPoint = {'ace': [[70, 0, 40], [70, 0, -40], [-70, 0, 40], [-70, 0, -40]],
                     'apple': [],
                     'cheezit': [[0, 80, 30], [0, 80, -30], [0, -80, 30], [0, -80, -30]],
                     'chococo': [],
                     'crayola': [[0, 80, 30], [0, -80, 30], [0, 80, -30], [0, -80, -30]],
                     'drill': [],
                     'genuine': [[60, 0, 60], [60, 0, -60], [-60, 0, 60], [-60, 0, -60]],
                     'moncher': [[70, 0, 40], [70, 0, -40], [-70, 0, 40], [-70, 0, -40]],
                     'mustard': [[0, 50, 30], [0, -50, -30], [0, -50, 30], [0, 50, -30]],
                     'papermate': [],
                     'scissors': [],
                     'tomatosoup': [],
                     'waffle': [[0, 80, 30], [0, -80, 30], [0, 80, -30], [0, -80, -30]],
                     'expo': []
                     }

NUM_CLASSES = len(CLASSES) # +1 for background

DETECTION_TH = 0.7

'''
network
'''
# if you do not assign the address ip, then the server will guess using command.
# # KIST
# KIST_STATIC_IP = '192.168.137.4'    # this and below address is required to know that which client is etri or kist.
# ETRI_STATIC_IP = '192.168.137.3'
# SERVER_IP = '192.168.137.50'        # '129.254.87.77'

# # ETRI - using Internet
# KIST_STATIC_IP = '129.254.87.77'
# ETRI_STATIC_IP = '129.254.87.77'
# SERVER_IP = '129.254.87.77'
# SERVER_PORT = 8020

# ETRI - using local connection
KIST_STATIC_IP = '192.168.0.50'
ETRI_STATIC_IP = '192.168.0.50'
SERVER_IP = '192.168.0.50'
SERVER_PORT = 8020

LEN_CHUNK = 10240                       # if the server cannot receive all data, then reduce the # of LEN_CHUNK.

'''
Pose estimation
'''
POSE_EST_TH_CORRESPONDENCES = 50        # default 20
POSE_EST_TH_INLIERS = 10                 # default 10


# # db for real & synth data - 3+1 OBJECTS for material property
# CLASSES = ('__background__', # always index 0
#            'mustard', 'airplane', 'cheezit')
#
# NUM_CLASSES = len(CLASSES) # +1 for background

# # # db for synthetic DATA
# # DB_LIST = ('__background__', # always index 0
# #                         'Ace', 'Apple', 'Champion', 'Cheezit', 'Chiffon',
# #                     'Chococo', 'Crayola', 'Cup', 'Drill', 'Expo',
# #                     'Genuine', 'Hammer', 'Highland', 'Mark', 'MasterChef',
# #                     'Moncher', 'Mustard', 'Papermate', 'Peg', 'Scissors',
# #                     'Sponge', 'TomatoSoup', 'Waffle', 'airplane', 'banana',
# #                     'strawberry') # change in n_classes in networks/VGGnetslsv1_train/test.py
#
# # # db for real data - 25 OBJECTS (+ sponge)
# # DB_LIST = ('__background__', # always index 0
# #            'strawberry', 'papermate', 'highland', 'genuine', 'mark',
# #            'expo', 'champion', 'orange', 'apple', 'cup',
# #            'banana', 'chiffon', 'crayola', 'scissors', 'tomatosoup',
# #            'drill', 'mustard', 'waffle', 'ace', 'airplane',
# #            'moncher', 'cheezit', 'chococo', 'sponge') # change n_classes in networks/VGGnetslsv1_train/test.py
#
# # db for real data - 23 OBJECTS (+ sponge)
# DB_LIST =  # change n_classes in networks/VGGnetslsv1_train/test.py

# for DBv6
# CLASSES = ( '__background__',
#             'ace', #1
#             'champion',
#             'cheezit',
#             'chiffon',
#             'chococo', #5
#             'crayola',
#             'expo',
#             'genuine',
#             'highland',
#             'mark', #10
#             'moncher',
#             'papermate',
#             'waffle',
#             'cup',
#             'drill',
#             'mustard',
#             'scissors',
#             'tomatosoup') # 18

# # for DBv7
# CLASSES = ( '__background__',
#                     'Ace', 'Apple', 'Champion', 'Cheezit', 'Chiffon',
#                     'Chococo', 'Crayola', 'Cup', 'Drill', 'Expo',
#                     'Genuine', 'Hammer', 'Highland', 'Mark', 'MasterChef',
#                     'Moncher', 'Mustard', 'Papermate', 'Peg', 'Scissors',
#                     'Sponge', 'TomatoSoup', 'Waffle', 'airplane', 'banana',
#                     'strawberry')
#
# Candidate_CLASSES = ('Ace', 'Cheezit', 'Chiffon',
#                     'Chococo', 'Crayola',
#                     'Genuine', 'Waffle')#'Drill',, 'airplane''Moncher','Mustard','TomatoSoup',
#
# # for DBV11_10obj
# CLASSES_10obj = ['__background__',
#                  'Ace', 'Apple', 'Cheezit', 'Chiffon', 'Crayola',
#                  'Drill', 'Genuine', 'Mustard', 'TomatoSoup', 'airplane']
# CLASSES = CLASSES_10obj

# for realV1
# CLASSES = ('__background__',
#            'strawberry', 'Papermate', 'Highland', 'Genuine', 'Mark',
#            'Expo', 'Champion', 'Orange', 'Apple', 'Cup',
#            'banana', 'Chiffon', 'Crayola', 'Scissors', 'TomatoSoup',
#            'Drill', 'Mustard', 'Waffle', 'Ace', 'airplane',
#            'Moncher', 'Cheezit', 'Chococo'
# )
#
# Candidate_CLASSES = (
# 'Ace','Apple', 'Champion', 'Cheezit', 'Chiffon',
# 'Chococo', 'Crayola','Cup', 'Drill', 'Expo', 'Genuine',
# 'Highland', 'Mark','Waffle', 'Moncher', 'Mustard', 'Papermate', 'Scissors', 'TomatoSoup'
# )
# # 'strawberry', 'airplane','Papermate', 'Orange', 'Apple', 'Cup','banana',  'Scissors', 'TomatoSoup','Drill', 'Mustard','Moncher',

# for real_sole + synthetic_duet
# # db for real data - 25 OBJECTS (+ sponge)
# CLASSES = ('__background__', # always index 0
#            'strawberry', 'papermate', 'highland', 'genuine', 'mark',
#            'expo', 'champion', 'orange', 'apple', 'cup',
#            'banana', 'chiffon', 'crayola', 'scissors', 'tomatosoup',
#            'drill', 'mustard', 'waffle', 'ace', 'airplane',
#            'moncher', 'cheezit', 'chococo', 'sponge') # change n_classes in networks/VGGnetslsv1_train/test.py