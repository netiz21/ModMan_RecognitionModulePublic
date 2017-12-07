# db for real & synth data - 22+1 OBJECTS exclude 'sponge', 'orange'
CLASSES = ('__background__', # always index 0
           'strawberry', 'papermate', 'highland', 'genuine', 'mark',
           'expo', 'champion', 'apple', 'cup',
           'banana', 'chiffon', 'crayola', 'scissors', 'tomatosoup',
           'drill', 'mustard', 'waffle', 'ace', 'airplane',
           'moncher', 'cheezit', 'chococo') # 'sponge'

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