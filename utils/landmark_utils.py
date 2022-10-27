import numpy as np
from skimage.morphology import binary_erosion
import cv2 as cv
import matplotlib.pyplot as plt


def mask2contour(mask):
    '''
    convert binary mask to contour points
    :param mask: binary mask
    :return: contour points
    '''
    mask = (mask > 0).astype(np.uint8)
    eroded_mask = binary_erosion(mask)
    contour = mask - eroded_mask
    return contour


def denoise(mask, kernel=10):
    '''
    using opening to denoise the mask
    :param mask: binary mask
    :return: denoised mask
    '''
    kernel = np.ones((kernel, kernel), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    # mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return mask


def get_points(mask):
    '''
    get the x,y-coordinates of the points with a given mask
    :param mask: binary mask
    :return: points: [(x1, y1), (x2, y2),....]
    '''
    points = []
    x, y = np.where(mask > 0)

    for i in range(len(x)):
        points.append((x[i], y[i]))
    return points
