import cv2
from utils import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("source", help="path untuk source image sebagai input.")
parser.add_argument("--maxdim", help="Argumen opsional untuk resize gambar. Dimensi terbesar dari gambar diresize menjadi ukuran ini. Gambar diresize dengan menjaga aspect ratio",type=int,default=640)
args=parser.parse_args()

if __name__ == "__main__":
    sourceimg = args.source
    maxdim = args.maxdim
    source_image = read_rgb(sourceimg)
    if source_image is None:
        print("Pastikan path valid.")
        exit(1)
    source_image = resize_ar(source_image,maxdim)
    # biar kalau gagak gak usah load dulu
    from face_detector import FaceDetector
    from segment import FaceParser
    fd = FaceDetector(template="./haar/haarcascade_frontalface_default.xml")
    fp = FaceParser('./bisenet/BiSeNet_keras.h5')
    showimg(source_image)

    face_image = fd.detect_crop2x(source_image.copy())
    # semua wajah yang terdeteksi di gambar diloop dan dicari masknya masing-masing
    for face in face_image:
        showimg(face)

        mask = fp.parse_one_face(face)
        showimg(mask)