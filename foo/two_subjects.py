from util import processimgs as pimg
import numpy as np
from scipy.spatial.distance import euclidean 
import sys
import os
from vggface import networks as vggn


if __name__=='__main__':
    imgpath1 = './datasets/feret-meb-vars/fa/00043/00043_931230_fa_var_0.jpeg'
    imgpath2 = './datasets/feret-meb-vars/fb/00043/00043_931230_fb_var_0.jpeg'
    imgpath3 = './datasets/feret-meb-vars/fa/00044/00044_931230_fa_var_0.jpeg'
    imgs = []
    imgs.append(pimg.load_image_plain(imgpath1)-[129.1863,104.7624,93.5940])

    imgs.append(pimg.load_image_plain(imgpath2)-[129.1863,104.7624,93.5940])
    imgs.append(pimg.load_image_plain(imgpath3)-[129.1863,104.7624,93.5940])
    
    network = vggn.VGGFaceVanilla()
    network.load_weights('./vggface/weights/plain-vgg-trained.ckpt')

    v1 = network.get_l2_vector([imgs[0],])
    v2 = network.get_l2_vector([imgs[1],])
    v3 = network.get_l2_vector([imgs[2],])
    print 'euclidean distance between same', euclidean(v1,v2)
    print 'euclidean distance between different', euclidean(v1,v3)
