import os
from util import processimgs as p

COLORFERET_PATH = './datasets/feret-color-images-only'
IMGCODE = 'fb'

if __name__ == "__main__":
    subjects = os.listdir(COLORFERET_PATH)

    for subj in subjects:
        crops_folder = os.path.join(COLORFERET_PATH, subj, 'vgg_crops')
        save_folder = os.path.join(COLORFERET_PATH, subj, IMGCODE+'_variations')
        img_filenames = filter(lambda x: IMGCODE in x,  os.listdir(crops_folder))

        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        
        print 'now working in folder', save_folder

        for img_file in img_filenames:
            p.generate_meb_variations_and_save(image_path=os.path.join(crops_folder, img_file),
                                               save_base_path=os.path.join(save_folder, img_file.split(".")[0]))
        
