import sys
sys.path.reverse() 
import numpy as np
import cv2

def load_image_plain(path):
    img = cv2.imread(path)

    if img is None:
        raise Exception("Image at path " + path + " not found. Check path or numpy, cv2, tensorflow import order.")
    else:
        return img

def crop_to_face(img):
    face_cascade = cv2.CascadeClassifier('/home/microway/opencv/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray, gray)  
    #detect faces. 
    faces = []
    reject_levels=[]
    face_cascade.detectMultiScale(gray, faces, reject_levels, scaleFactor=1.05, minNeighbors=5, flags=0, outputRejectLevels=True)
    tmp=len(reject_levels)
    #If none found use entire image
    if tmp==0:
        img2=img
    else:
        #use the last face detected since it is the biggest
        tmp=np.argmax(reject_levels)
        length=faces[tmp][2]
        #make crop size a bit bigger than the detection 
        offset=round(length*0.05)
        x1=max(0,faces[tmp][1]-offset)
        y1=max(0,faces[tmp][0]-offset)
        x2=min(faces[tmp][1]+faces[tmp][3]+offset,img.shape[0])
        y2=min(faces[tmp][0]+faces[tmp][2]+offset,img.shape[1])
        img2=img[x1:x2,y1:y2]

    return img2


def load_crop_adjust(image_path, width=224, height=224):
    '''
    loads a single image, crops it, normalizes it, resizes it to VGGFace specs,
    and returns it as a NumPy array 
    '''
    img = load_image_plain(image_path)
    img2 = crop_to_face(img)
        
    img2 = img.astype(np.float32)

    #subtract the average so that all pixels are close to the average
    #img2 -= [129.1863,104.7624,93.5940]

    # have to resize it here to comply with VGGFace specs
    img2 = cv2.resize(img2, (width, height))
    img2 = np.array([img2,])  
    return img2

def load_crop_adjust_save(image_path, save_path, width=224, height=224):
    img = load_crop_adjust(image_path, width, height)[0]
    cv2.imwrite(save_path, img)

def load_generate_meb_variations(image_path, crop_w=196, crop_h=196):
    '''
    loads an image, generates all crops of the specified size
    and then returns an image OF THE SAME SIZE 
    (i.e. the image is resized back to its original)
    '''
    img = load_image_plain(image_path)
    orig_h, orig_w, orig_c = img.shape #img is a numpy array, 3rd value is channels
    
    final_imgs = []

    i, j = 0, 0

    while (crop_w + i) < orig_w:
        while (crop_h + j) < orig_h:
            final_imgs.append(cv2.resize(img[i:crop_w+i, j:crop_h+j], (orig_w, orig_h)))
            j = j+4
        j = 0
        i = i+4

    final_imgs.extend(map(lambda x: cv2.flip(x, flipCode=1), final_imgs))
    return final_imgs

def generate_meb_variations_and_save(image_path, save_base_path, 
                                      crop_w=196, crop_h=196,
                                      extension='.ppm'):
    imgs = load_generate_meb_variations(image_path, crop_w, crop_h)
    for i in xrange(len(imgs)):
        cv2.imwrite(save_base_path+'_var_'+str(i)+extension, imgs[i])
