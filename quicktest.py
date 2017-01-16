from util import processimgs as pimg
import cv2
img= pimg.load_crop_adjust_save("datasets/feret-color-images-only/00070/00070_940928_fa.ppm", "testimg.jpg")
print img
