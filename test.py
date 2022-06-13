import cv2
import numpy as np
from pcn.utils import Window
from pcn.models import load_model
from pcn.pcn import pcn_detect
import time

def crop_face(img, face:Window, crop_size=200):
    x1 = face.x
    y1 = face.y
    x2 = face.width + face.x - 1
    y2 = face.width + face.y - 1
    centerX = (x1 + x2) // 2
    centerY = (y1 + y2) // 2
    lst = (x1, y1), (x1, y2), (x2, y2), (x2, y1)
    pointlist = [rotate_point(x, y, centerX, centerY, face.angle) for x, y in lst]
    srcTriangle = np.array([
        pointlist[0],
        pointlist[1],
        pointlist[2],
    ], dtype=np.float32)
    dstTriangle = np.array([
        (0, 0),
        (0, crop_size - 1),
        (crop_size - 1, crop_size - 1),
    ], dtype=np.float32)
    rotMat = cv2.getAffineTransform(srcTriangle, dstTriangle)
    ret = cv2.warpAffine(img, rotMat, (crop_size, crop_size))
    return ret, pointlist


def rotate_point(x, y, centerX, centerY, angle):
    x -= centerX
    y -= centerY
    theta = -angle * np.pi / 180
    rx = int(centerX + x * np.cos(theta) - y * np.sin(theta))
    ry = int(centerY + x * np.sin(theta) + y * np.cos(theta))
    return rx, ry


def crop(img, winlist, size=200):
    """
    Returns:
        list of [face, location] 
    """
    faces = list(map(lambda win: crop_face(img, win, size), winlist))
    return faces


def get_face(img, nets, is_crop=False):
    winlist = pcn_detect(img, nets) 
    if is_crop:
        faces = crop(img, winlist)
        faces = [f[0] for f in faces]
        img = np.hstack(faces)
    return img

start = time.time()
nets = load_model()
img = cv2.resize(cv2.imread('nt.jpg'), (224,224))



