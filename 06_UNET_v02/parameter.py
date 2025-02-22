import os
#PARAMETROS

TEST_SELECTED_INDEX = 1
SAVE_PLOTS = True
SAVE_MODEL = True

USE_PRETRAINED_MODEL = True  # Set to True to use a pre-trained model instead of training a new one
RE_TRAIN_MODEL = False        # Set to True if you want to retrain a pre-trained model

TEST_AVAILABLE = ["EPFL - Mitocondria Electron Microscopy",
                  "Chest CT Segmentation"] 
#CT-CHEST\MARCOPOLO
BLOCK_ID = ["ID00035637202182204917484", 
            "ID00027637202179689871102", 
            "ID00139637202231703564336",
            "ID00014637202177757139317",
            "ID00032637202181710233084",
            "ID00426637202313170790466"]

#VARIABLE
AUGMENTATION = True
N_AUGMENTATION = 3
RESIZE = False
RESIZE_VALUE = (256,256)
RATIO = 0.5
CHOP_VALUE = 1000
EPOCHS = 10
BATCH_SIZE = 30
LEARNING_RATE = 3e-4
SHUFFLE = True  
MULTI_CLASS = False

#RESULT
RESULT_DIR = os.path.join("/mnt", "workspace", "cmorenor", "RESULT")
PATH_DATASET = os.path.join('/mnt','researchers','marcelo-andia','datasets')
PATH_CT_MARCOPOLO = os.path.join(PATH_DATASET,'CT-Chest', 'Marco Polo')
PATH_EPFL = os.path.join(PATH_DATASET,'EPFL')

