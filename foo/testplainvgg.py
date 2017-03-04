from util import processimgs as pimg
from vggface import plainvgg as pvgg

vgg_weights = './vggface/weights/plain-vgg-trained.ckpt'
img_path = './datasets/feret-meb-vars/fa/00070/00070_931230_fa_var_0.jpeg'


#network1 = pvgg.VGGFaceWithL2("/gpu:3")
#network1.load_weights(vgg_weights)
#print 'withl2:', ','.join(map(str,network1.forwardprop([pimg.load_adjust_avg(img_path),]))) 

network2 = pvgg.VGGFaceWithoutL2("/gpu:3")
network2.load_weights(vgg_weights)
print 'nol2:', ','.join(map(str,network2.forwardprop([pimg.load_adjust_avg(img_path),]))) 
