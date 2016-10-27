import os
from util import processimgs as p

COLORFERET_PATH = './datasets/feret-color-images-only'
SAVE_PATH='./datasets/feret-meb-vars/'
IMGCODE = 'fb'

if __name__ == "__main__":
    base_save_folder = os.path.join(SAVE_PATH, IMGCODE)

    with open('fa.txt','r') as f:
        for line in f:
            subj, img_file = line.strip('\n').split(' ')
            subjid, date, pose = img_file.split('_', 2)
            crops_folder = os.path.join(COLORFERET_PATH, subj, 'vgg_crops')

            if not os.path.exists(crops_folder):
                continue

            save_folder=os.path.join(base_save_folder, subj)

            if not os.path.exists(save_folder):
               os.mkdir(save_folder) 

            print 'now processing subject', subj

            p.generate_meb_variations_and_save(image_path=os.path.realpath(os.path.join(crops_folder, img_file)),
                                               save_base_path=os.path.join(save_folder, 
                                                                           img_file.split(".")[0]),
                                               extension='.jpeg')
