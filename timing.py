from time import time
import numpy as np
import cv2
from tqdm import tqdm
from utils import *

testimg = read_rgb('./samples/2.jpg')
testimg = cv2.resize(testimg,(640,480),interpolation=cv2.INTER_AREA)

from face_detector import FaceDetector
fd = FaceDetector(template="./haar/haarcascade_frontalface_default.xml")
timings = np.zeros(1001,dtype=np.float64)
print("Measuring time for detection..")
for i in tqdm(range(1001)):
    start = cv2.getTickCount()
    detect_crop = fd.detect2x(testimg)
    elapsed = cv2.getTickCount() - start
    elapsed /= cv2.getTickFrequency()
    timings[i] += elapsed
print(f"First detection took : {timings[0]} seconds. This is detection time + loading time.")
timings = timings[1:]
mean= timings.mean()
std = timings.std()
minimum = timings.min()
maximum = timings.max()
print(f"Head-shoulders detection took\nAvg : {mean} seconds\nStdev : {std} seconds\nSpanning {minimum} to {maximum} seconds.")

head_shoulders = fd.detect_crop2x(testimg)[0]
timings = np.zeros(1001,dtype=np.float64)
from segment import FaceParser
fp = FaceParser('./bisenet/BiSeNet_keras.h5')
print("Measuring segmentation time..")
for i in tqdm(range(1001)):
    start = cv2.getTickCount()
    segment = fp.parse_one_face(head_shoulders)
    elapsed = cv2.getTickCount() - start
    elapsed /= cv2.getTickFrequency()
    timings[i] += elapsed
print(f"First detection took : {timings[0]} seconds. This is detection time + loading time")
timings = timings[1:]
mean= timings.mean()
std = timings.std()
minimum = timings.min()
maximum = timings.max()
print(f"Head-shoulders segmentation took\nAvg : {mean} seconds\nStdev : {std} seconds\nSpanning {minimum} to {maximum} seconds.")
