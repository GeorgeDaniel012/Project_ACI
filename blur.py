import cv2
import numpy as np

def gaussian_blur(img, ksize=15, sigma=0):
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def motion_blur(img, length=25, angle=0):
    if length % 2 == 0: # obligatoriu impar
        length += 1
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    center = (length / 2 - 0.5, length / 2 - 0.5)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)
    