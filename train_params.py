EVAL_SET_BASE = './datasets/feret-meb-vars/fb'
TRAIN_SET_BASE = './datasets/feret-meb-vars/fa'
PATH_STOM = './datasplits/subjtomeb_colorferet.json'
PATH_FTOS =  './datasplits/fa_filepath_to_subject_colorferet.json'
#VGG_WEIGHTS_PATH = './vggface/weights/plain-vgg-trained.ckpt'
#VGG_WEIGHTS_PATH = 'output/euclidean_32_subjects/weights/weights_epoch_4_final.ckpt'
VGG_WEIGHTS_PATH = './output/bigger_batch_32_subjects/weights'
SAVE_FOLDER = './output/bigger_batch_32_subjects'
#SAVE_FOLDER = './output/euclidean_32_subjects_cont/'
FILES = ['weights_epoch_9_final.ckpt','weights_epoch_16_final.ckpt','weights_epoch_13_final.ckpt']
EVAL_SAMPLE_SIZE = 196 # number of images per subject used to evaluate the network 
NUM_EVAL_ITERS = 2 #how many times should the accuracy measures be computed with randomized imposters
EPOCHS = 20 
BATCH_SIZE = 4*49
LEARNING_RATE = 0.01
CHECKPOINT = 8
SUBJ_FOR_TRAINING = sorted(os.listdir(TRAIN_SET_BASE))[:500]
