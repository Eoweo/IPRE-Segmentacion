import os
#PARAMETROS

TEST_SELECTED_INDEX = 0
SAVE_PLOTS = True
SAVE_MODEL = True
EARLY_STOP = False

USE_PRETRAINED_MODEL = False  # Set to True to use a pre-trained model instead of training a new one
RE_TRAIN_MODEL = False        # Set to True if you want to retrain a pre-trained model
WORKERS = 8

TEST_AVAILABLE = ["EPFL - Mitocondria Electron Microscopy",
                  "Chest CT Segmentation"] 
#CT-CHEST\MARCOPOLO
BLOCK_ID = ["ID00035637202182204917484", 
            "ID00027637202179689871102", 
            "ID00139637202231703564336",
            "ID00014637202177757139317",
            "ID00032637202181710233084",
            "ID00426637202313170790466",
            "ID00075637202198610425520",
            "ID00104637202208063407045",
            "ID00123637202217151272140",
            "ID00109637202210454292264",
            "ID00089637202204675567570"]
#ID00104637202208063407045 -> mascaras estan al revés que imagenes osea imagen 0 -> mask 497
#ID00075637202198610425520 -> mascaras estan al revés que imagenes
#ID00426637202313170790466 -> mascaras estan al revés que imagenes
#ID00123637202217151272140
#ID00109637202210454292264
#ID00089637202204675567570
#VARIABLE

AUGMENTATION = True
RESIZE = False
RESIZE_VALUE = (512,512)
RATIO = 0.8

CHOP_DATA = False
CHOP_DATA_VALUE = 10000

CHOP_PATIENT = False
CHOP_PATIENT_VALUE = 8

EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
SHUFFLE = True  
MULTI_CLASS = False

#RESULT
RESULT_DIR = os.path.join("/mnt", "workspace", "cmorenor", "RESULT")
PATH_DATASET = os.path.join('/mnt','researchers','marcelo-andia','datasets')
PATH_CT_MARCOPOLO = os.path.join(PATH_DATASET,'CT-Chest', 'Marco Polo')
PATH_CT_IMAGE = os.path.join(PATH_CT_MARCOPOLO, "archive", "images", "images")
PATH_CT_MASK =  os.path.join(PATH_CT_MARCOPOLO, "archive", "masks", "masks")
PATH_EPFL = os.path.join(PATH_DATASET,'EPFL')
INFERENCE_PATH = os.path.join(RESULT_DIR, "..", "INFERENCE")

