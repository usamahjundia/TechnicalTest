import cv2
import numpy as np
from utils import boundBoxes

class FaceDetector:
    def __init__(self, template):
        self.template = template
        self.ready = False
        self.detector = cv2.CascadeClassifier()
        self.load()
    
    def load(self):
        self.ready = self.detector.load(self.template)
    
    def detect(self,image,equalize=False):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if equalize:
            image = cv2.equalizeHist(image)
        boxes = self.detector.detectMultiScale(image)
        if boxes == tuple():
            return []
        return boxes
    
    def detect2x(self,image,equalize=False):
        boxes = self.detect(image,equalize)
        if len(boxes) == 0:
            return []
        boxes[:,:2] -= boxes[:,2:]//2
        boxes[:,2:] *= 2
        shape = image.shape
        return boundBoxes(boxes,shape[1],shape[0],xywh=True)
        
    def detect_crop(self,image,equalize=False):
        boxes = self.detect(image,equalize)
        faces = []
        for (x,y,w,h) in boxes:
            face = image[y:y+h,x:x+w]
            faces.append(face)
        return faces
    
    def detect_crop2x(self,image,equalize=False):
        boxes = self.detect2x(image,equalize)
        faces = []
        for (x,y,w,h) in boxes:
            face = image[y:y+h,x:x+w]
            faces.append(face)
        return faces
    
    def detect_draw(self,image,equalize=False,thickness=1):
        boxes = self.detect(image,equalize)
        ret_img = self.draw(image,boxes,thickness)
        return ret_img
    
    def draw(self,image,boxes,thickness=1):
        img = image.copy()
        for (x,y,w,h) in boxes:
            xmax = x + w
            ymax = y + h
            cv2.rectangle(img,(x,y),(xmax,ymax),(0,255,255),thickness)
        return img