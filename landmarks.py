import math
import os
import torch
import numpy as np
from utils.landmark_utils import mask2contour, denoise, get_points
import cv2 as cv
import matplotlib.pyplot as plt
from inference import inference
from numpy.polynomial import Polynomial
from tqdm import tqdm

'''
The methods for landmarks locating
Run this file to test the alignment. the test image and mask 
'''


def locate_anatomical_axis(mask):
    '''
    :param mask: find anatomical_axis of femur or tibia by least-square fitting the mid part of the bone shaft contour
    :return: points of the anatomical axis line
    '''
    # set the percentage to cut off at the top and bottom of the bone area
    cut_out = 0.1
    # transfer mask to contour
    contour = mask2contour(mask)
    pixels = np.where(contour > 0)
    x, y = pixels
    num_cut_point = int(len(x) * cut_out)
    # sort the pixels
    idx = np.argsort(x)
    pixels_sort = (x[idx], y[idx])
    # cut the top and bottom piexls
    pixels_filtered = (pixels_sort[0][num_cut_point:-num_cut_point], pixels_sort[1][num_cut_point:-num_cut_point])
    # fit line
    line = Polynomial.fit(
        pixels_filtered[0],
        pixels_filtered[1],
        1,
        domain=[pixels[0].min(), pixels[0].max()],
    )
    return (pixels[0], line(pixels[0]))


def locate_femur_head(mask):
    '''
    locate the center of femur head by fit a circle to femur head area
    :param mask: the rgb mask of unilateral long-leg x-ray
    :return: the center of femur head x_c, y_c, and radius of the circle
    '''
    pixels = np.where(mask > 0)
    # find the pixel bound of the femur
    max_x, max_y = np.max(pixels[0]), np.max(pixels[1])
    min_x, min_y = np.min(pixels[0]), np.min(pixels[1])
    # narrow down the region of femur head
    region_x = (min_x - 10, int(min_x + 0.15 * (max_x - min_x)))
    region_y = (int(min_y + 0.35 * (max_y - min_y)), max_y)
    # get the pixels in region of femur head
    region_mask = mask[region_x[0]:region_x[1] + 1, region_y[0]:region_y[1] + 1].copy()
    region_mask[:, :2] = 0
    # plot the region of interest (femur head)
    # plt.imshow(region_mask)
    # plt.show()
    dist_map = cv.distanceTransform(region_mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    _, radius, _, center = cv.minMaxLoc(dist_map)
    y_c, x_c = center
    y_c = y_c + region_y[0]
    x_c = x_c + region_x[0]
    return (x_c, y_c), int(radius)


def locate_knee_center(mask):
    '''
    locate the center of the knee
    :param mask: the femur mask. should be a binary mask
    :return: the coordinates of the center of the knee
    '''
    # contour = mask2contour(mask)
    contour = np.where(mask > 0)
    contour_points = set()
    for i in range(len(contour[0])):
        contour_points.add((contour[0][i], contour[1][i]))

    femur_aa = locate_anatomical_axis(mask)
    aa_points = set()
    for i in range(len(femur_aa[0])):
        aa_points.add((femur_aa[0][i], int(femur_aa[1][i])))
        aa_points.add((femur_aa[0][i], round(femur_aa[1][i])))

    center = (0, 0)
    for i in list(aa_points.intersection(contour_points)):
        if i[0] > center[0]:
            center = i
    return center


def locate_tibia_center(mask):
    '''
    locate the center of the tibia, and the center of the ankle
    :param mask: the tibia mask. should be a binary mask
    :return: the coordinates of the center of the tibia and the center of the ankle
    '''
    # contour = mask2contour(mask)
    contour = np.where(mask > 0)
    contour_points = set()
    for i in range(len(contour[0])):
        contour_points.add((contour[0][i], contour[1][i]))

    tibia_aa = locate_anatomical_axis(mask)
    aa_points = set()
    for i in range(len(tibia_aa[0])):
        aa_points.add((tibia_aa[0][i], int(tibia_aa[1][i])))

    tibia_center = (1024, 1024)
    ankle_center = (0, 0)
    for i in list(aa_points.intersection(contour_points)):
        if i[0] < tibia_center[0]:
            tibia_center = i
        if i[0] > ankle_center[0]:
            ankle_center = i
    return tibia_center, ankle_center


def calculate_hka(femur_head, knee_center, tibia_center, ankle_center):
    '''
    calculate the hka angle. Inputs are the coordinates of the landmarks
    :return: the hka angle
    '''
    a, b, c, d = np.array(femur_head), np.array(knee_center), np.array(tibia_center), np.array(ankle_center)
    vector_femur = a - b
    vector_tibia = c - d
    dot_product = np.dot(vector_femur, vector_tibia)
    mod = np.linalg.norm(vector_femur) * np.linalg.norm(vector_tibia)
    angle = math.degrees(math.acos(dot_product / mod))
    if np.cross(vector_femur, vector_tibia) < 0:
        angle = angle
    else:
        angle = - angle
    return angle


def locate_mechanical_axis(mask):
    '''
    locate all landmarks of femur mechanical axis and tibia mechanical axis
    :param mask: a predicted RGB mask
    :return: the coordinates of the center of the femur head, the center of the knee
    the center of the tibia, the center of the ankle and the hka value
    '''
    femur_mask = mask[:, :, 1]
    femur_mask[700:, :] = 0
    femur_mask = denoise(femur_mask, kernel=10)

    tibia_mask = mask[:, :, 0]
    tibia_mask[:300, :] = 0
    tibia_mask = denoise(tibia_mask, kernel=10)

    femur_head, r = locate_femur_head(femur_mask)
    # femur_aa = locate_anatomical_axis(femur_mask)
    # tibia_aa = locate_anatomical_axis(tibia_mask)

    knee_center = locate_knee_center(femur_mask)
    # knee_center = locate_joint_center(femur_mask)
    tibia_center, ankle_center = locate_tibia_center(tibia_mask)
    # ankle_center = locate_joint_center(tibia_mask)
    hka = calculate_hka(femur_head, knee_center, tibia_center, ankle_center)

    return femur_head, r, knee_center, tibia_center, ankle_center, hka


def draw_mechanical_axis(img, mask, img_name=None, save_dir=None):
    '''
    visualize the segmentation and alignment results
    :param img: x-ray image
    :param mask: the predicted RGB mask
    :param img_name: the name of the img.
    :param save_dir: the fold to save the results
    :return:
    '''

    femur_head, r, knee_center, tibia_center, ankle_center, hka = locate_mechanical_axis(mask)
    mask[:, :, 2] = 0

    dpi = 80
    height, width, depth = img.shape
    figsize = width / float(dpi), height / float(dpi)
    plt.figure(figsize=figsize)
    plt.axis("off")

    # plot image
    plt.imshow(img)
    # plot mask
    plt.imshow(mask, cmap='jet', alpha=0.2)

    # plot femur head center
    circle = plt.Circle((femur_head[1], femur_head[0]), r, color='y', fill=False)
    plt.gca().add_patch(circle)
    plt.plot(femur_head[1], femur_head[0], ",", color="y")
    # plot femur anatomical axis
    # plt.plot(femur_aa[1], femur_aa[0], "-",color="y", linewidth=1.0)
    # plot tibia anatomical axis
    # plt.plot(tibia_aa[1], tibia_aa[0], "-",color="y", linewidth=1.0)
    # knee center
    plt.plot(knee_center[1], knee_center[0], ",", color="y")
    # tibia center
    plt.plot(tibia_center[1], tibia_center[0], ",", color="y")
    # ankle center
    plt.plot(ankle_center[1], ankle_center[0], ",", color="y")
    # plot mechanical axis
    plt.plot([femur_head[1], knee_center[1]], [femur_head[0], knee_center[0]], "-", color="y", linewidth=1.0)
    plt.plot([tibia_center[1], ankle_center[1]], [tibia_center[0], ankle_center[0]], "-", color="y", linewidth=1.0)
    plt.text(3, 160, 'HKA = {:.2f}'.format(hka), color='y', size=12)
    plt.savefig(save_dir + img_name, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    plt.close('all')
    # plt.show()


if __name__ == '__main__':
    img_path = "./data/inference/images/"
    mask_path = "./data/inference/results/"
    save_path = "./data/inference/alignment/"
    model_path = "exp_1_128x1024_dc_model.pth"

    model = torch.load(model_path)
    inference(img_path, model)

    img_names = os.listdir(img_path)
    for name in tqdm(img_names):
        img = cv.imread(img_path + name)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        mask = cv.imread(mask_path + name[:-4] + '.png')
        mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)

        draw_mechanical_axis(img, mask, img_name=name, save_dir=save_path)
