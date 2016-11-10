EVAL_SET_BASE = './datasets/feret-meb-vars/fb'
TRAIN_SET_BASE = './datasets/feret-meb-vars/fa'
PATH_STOM = './datasplits/subjtomeb_colorferet.json'
PATH_FTOS =  './datasplits/fa_filepath_to_subject_colorferet.json'
VGG_WEIGHTS_PATH = './vggface/weights/plain-vgg-trained.ckpt'
SAVE_FOLDER = '/home/evtimovi/neural-nets-thesis/output/train_two_subjects_nov10'
EVAL_SAMPLE_SIZE = 196 # number of images per subject used to evaluate the network 
EPOCHS = 2
BATCH_SIZE = 49
LEARNING_RATE = 0.001
CHECKPOINT = 8
SUBJ_FOR_TRAINING = ['00044', '00043']

