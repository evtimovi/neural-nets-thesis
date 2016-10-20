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
    faces = face_cascade.detectMultiScale3(gray, 1.05, 5, outputRejectLevels=True)
    tmp=len(faces[2])
    #If none found use entire image
    if tmp==0:
        img2=img
    else:
        #use the last face detected since it is the biggest
        tmp=np.argmax(faces[2])
        length=faces[0][tmp][2]
        #make crop size a bit bigger than the detection 
        offset=round(length*0.05)
        x1=max(0,faces[0][tmp][1]-offset)
        y1=max(0,faces[0][tmp][0]-offset)
        x2=min(faces[0][tmp][1]+faces[0][tmp][3]+offset,img.shape[0])
        y2=min(faces[0][tmp][0]+faces[0][tmp][2]+offset,img.shape[1])
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
    img2 -= [129.1863,104.7624,93.5940]

    # have to resize it here to comply with VGGFace specs
    img2 = cv2.resize(img2, (width, height))
    img2 = np.array([img2,])  
    return img2

