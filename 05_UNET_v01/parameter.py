import os
#PARAMETROS

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
ROTATION = True
RESIZE = True
RESIZE_VALUE = (256,256)
CHOP_VALUE = None
EPOCHS = 20
BATCH_SIZE = 8
SHUFFLE = True  
MULTI_CLASS = False

#RESULT
RESULT_DIR = os.path.join("/mnt", "workspace", "cmorenor", "RESULT")
PATH_DATASET = os.path.join('/mnt','researchers','marcelo-andia','datasets')
PATH_CT_MARCOPOLO = os.path.join(PATH_DATASET,'CT-Chest', 'Marco Polo')
PATH_EPFL = os.path.join(PATH_DATASET,'EPFL')

