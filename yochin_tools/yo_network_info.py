PATH_BASE = '/home/jang/smallcorgi_Faster-RCNN_TF_yochin'

# db for real & synth data - 22+1 OBJECTS exclude 'sponge', 'orange'
CLASSES = ('__background__', # always index 0
           'strawberry', 'papermate', 'highland', 'genuine', 'mark',
           'expo', 'champion', 'apple', 'cup',
           'banana', 'chiffon', 'crayola', 'scissors', 'tomatosoup',
           'drill', 'mustard', 'waffle', 'ace', 'airplane',
           'moncher', 'cheezit', 'chococo') # 'sponge'

Candidate_CLASSES = (
    'ace', 'airplane', 'drill', 'mustard', 'waffle',
    'chiffon', 'crayola', 'genuine', 'tomatosoup', 'cheezit')
    #'strawberry', 'papermate', 'highland', 'mark',
           # 'expo', 'champion', 'apple', 'cup',
           # 'banana', 'scissors',
           # 'moncher', 'chococo')

NUM_CLASSES = len(CLASSES) # +1 for background



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