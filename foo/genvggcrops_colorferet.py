import sys
import os
import random
from util import processimgs as p

COLORFERET_PATH = './datasets/feret-color-images-only'

def create_crops(imgdict, dir_name):
    for subjdir in imgdict.keys():
        print 'now processing directory', subjdir
        save_directory = os.path.join(subjdir, dir_name)
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
        p.load_crop_adjust_save(image_path=os.path.join(subjdir,imgdict[subjdir]),
                                save_path=os.path.join(save_directory, imgdict[subjdir]))


if __name__=="__main__":
#    fas_all = {}
#    fbs_all = {}

    subjects = os.listdir(COLORFERET_PATH)
    subjects = filter(lambda s: s.startswith('0'), subjects)

    for subj in subjects:
        subj_folder=os.path.join(COLORFERET_PATH, subj)
        print 'subject folder', subj_folder

        crops_folder = os.path.join(COLORFERET_PATH, subj, 'vgg_crops')

        fas = filter(lambda x: 'fa' in x and not x.startswith('fa'),  os.listdir(subj_folder))
        fbs = filter(lambda x: 'fb' in x and not x.startswith('fb'),  os.listdir(subj_folder))

        if not os.path.exists(crops_folder):
            os.mkdir(crops_folder)

#        if len(fas) > 1 and len(fbs) > 1:
        for imfa in fas:
            p.load_crop_adjust_save(image_path=os.path.join(subj_folder, imfa),
                                    save_path=os.path.join(crops_folder, imfa))
        for imfb in fbs:
            p.load_crop_adjust_save(image_path=os.path.join(subj_folder, imfb),
                                    save_path=os.path.join(crops_folder, imfb))


#    for dirpath, dirnames, filenames in os.walk('./datasets/feret-color-images-only'):
#        fas = filter(lambda x: x.split('_',3)[2] == 'fa' if len(x.split('_',3)) > 3 else False, filenames) 
#        fbs = filter(lambda x: x.split('_',3)[2] == 'fb' if len(x.split('_',3)) > 3 else False, filenames)
#
#        if len(fas) > 0 and len(fbs) > 0:
#            fas_all[dirpath] = fas[0]
#            fbs_all[dirpath] = fbs[0]
#       
#    dir_name = 'vgg_crops'
#
#    create_crops(fas_all, dir_name)
#    create_crops(fbs_all, dir_name)
