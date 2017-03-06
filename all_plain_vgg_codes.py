from util import processimgs as pimg
import os
import sys
from vggface import plainvgg as pvgg

vgg_weights = './vggface/weights/plain-vgg-trained.ckpt'
img_base_path = './datasets/feret-meb-vars/'

execfile("subjects.input")
subjects = SUBJ_FOR_TRAINING

network2 = pvgg.VGGFaceWithoutL2("/gpu:3")
network2.load_weights(vgg_weights)

def get_top_img_path(code, subj):
    subj_dir = os.path.join(img_base_path,code,subj)
    all_imgs = sorted(os.listdir(subj_dir))
    if len(all_imgs) < 1:
        sys.stderr.write(subj_dir+"is empty\n")
        return None
    return os.path.join(subj_dir,all_imgs[0])

def do_img(f, subject, code, img_path):
    f.write(s
            + "," 
            + img_path[len(img_base_path)+9:]
            + ","
            + code
            + ","
            +",".join(map(str, network2.forwardprop([pimg.load_adjust_avg(img_path),]))) 
            + "\n")


with open('vgg_output_one_img_per_type_per_subject.csv', 'w') as f:
    f.write("subject,img,img_code,code_begin\n")
    for s in subjects:
        fa_img_path = get_top_img_path('fa',s)
        if fa_img_path is not None:
            do_img(f, s, "fa", fa_img_path)

        fb_img_path = get_top_img_path('fb',s)
        if fb_img_path is not None:
            do_img(f, s, "fb", fb_img_path)

        rc_img_path = get_top_img_path('rc',s) 
        if rc_img_path is not None:
            do_img(f, s, "rc", rc_img_path)
