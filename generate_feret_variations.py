import os
import sys
from util import processimgs as pimg

if __name__=="__main__":
    origin_base = "./datasets/feret-color-images-only/"
    destination_base = "./datasets/feret-meb-vars/" 
    fpath = raw_input("provide the name of the file:")
    prefix = fpath.split(".")[0]
    prefix = prefix[(len(prefix)-2):]

    dest_folder = os.path.join(destination_base, prefix)
    
    if os.path.exists(dest_folder):
        if len(os.listdir(dest_folder)) > 0:
            sys.stderr.write("The destination folder "+dest_folder +" is already full. Please, delete all subject folders.\n") 
            sys.exit(1)
    else:
        os.mkdir(dest_folder)
    
    subjects = []

    with open(fpath, 'r') as f:
        for l in f:
            subjid, imgbase, captdate, captgall, daysdiff = l.split(" ")
            if int(subjid) < 43:
                continue
            subjects.append(subjid)
            subj_path_dest = os.path.join(dest_folder, subjid)
            if not os.path.exists(subj_path_dest):
                os.mkdir(subj_path_dest)
            
            imgpath_orig = os.path.join(origin_base, subjid, imgbase)
            imgname_dest = imgbase.split(".")[0] + ".jpg"
            imgpath_dest = os.path.join(subj_path_dest, imgname_dest)

            sys.stdout.write("now cropping subject " + subjid + "\n")
            sys.stdout.flush()
            # first crop it 
            pimg.load_crop_adjust_save(imgpath_orig, imgpath_dest)

            sys.stdout.write("now generating variations for subject " + subjid + "\n")
            sys.stdout.flush()
            # then do the MEB variations
            pimg.generate_meb_variations_and_save(imgpath_dest, subj_path_dest)  

    subjects_file = os.path.join('datasplits', prefix+'_subjects.txt')
    sys.stdout.write("Done. Writing subjects to file " + subjects_file + "\n")
    with open(subjects_file, 'w') as f:
        f.write(subjects)
