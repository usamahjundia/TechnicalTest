import cv2
import matplotlib.pyplot as plt
import numpy as np

def showimg(img,figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def read_rgb(impath):
    img = cv2.imread(impath)
    if img is None:
        return None
    return img[:,:,::-1]

def resize_ar(image,maxdim):
    shape = image.shape
    h = shape[0]
    w = shape[1]
    ar = h/w
    if w > h:
        scalefactor = maxdim/w
        h*=scalefactor
        h= int(h)
        w = maxdim
    else:
        scalefactor = maxdim/h
        w*=scalefactor
        w = int(w)
        h = maxdim
    if h*w > shape[0]*shape[1]:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_AREA
    return cv2.resize(image,(w,h),interpolation)

def drawLandmarks(frame, landmarks):
    for x,y in landmarks:
        cv2.circle(frame,(x,y),2,(0,0,255),thickness=-1)

def drawboxes(frame,boxes,thickness=1):
    img = frame.copy()
    for x,y,xm,ym in boxes:
        cv2.rectangle(img,(x,y),(xm,ym),(0,255,0),thickness)
    return img

def boundBoxes(boxes,imw,imh,xywh=False):
    if xywh:
        boxes[:,2:] += boxes[:,:2]
    boxes[:,0] = np.maximum(boxes[:,0],0)
    boxes[:,1] = np.maximum(boxes[:,1],0)
    boxes[:,2] = np.minimum(boxes[:,2],imw-1)
    boxes[:,3] = np.minimum(boxes[:,3],imh-1)
    if xywh:
        boxes[:,2:] -= boxes[:,:2]
    return boxes

def getFaceOnly(image,facemap):
    mask = np.isin(facemap,FACE)
    mask = np.repeat(mask[...,None],3,axis=-1)
    return image * mask

def getminbbox(pts):
    xs = pts[:,0]
    ys = pts[:,1]
    return xs.min(), ys.min(), xs.max(), ys.max()

def pointsToNumpy(points):
    n = points.num_parts
    ret = np.zeros((n,2),dtype=np.int32)
    for i in range(n):
        temp = points.part(i)
        ret[i] = [temp.x,temp.y]
    return ret