import cv2 as cv
import json
import os
import numpy as np
from tqdm import tqdm

'''
Run this script to convert .json annotation to .png mask images.
Modified on https://github.com/rukon-uddin/Json-annotations-to-mask-images,
The original implementation can only generate a binary mask. This implementation 
extends to a 3-channel mask. 

The background is assigned in blue channel, the femur is assigned in green channel, and
the tibia is assigned in red channel.

It is implemented specifically for COMP8604 project, thus it only works for the dataset
involved in COMP8604 project.

'''

dataset_dir = "../data/1_preprocessed/images/"
save_dir = "../data/1_preprocessed/masks/"

img_path = []
label_path = []
for i in os.listdir(dataset_dir):
    if (i.endswith(".json")):
        label_path.append(dataset_dir + "/" + i)
    else:
        img_path.append(dataset_dir + "/" + i)

for c, i in tqdm(enumerate(label_path), total=len(label_path)):
    # create a mask with blue background
    img = cv.imread(img_path[c])
    h, w, _ = img.shape
    mask = np.zeros((h, w, 3))
    mask[:, :, 0] = 255
    with open(i) as f:
        data = json.load(f)
    # get image name without postfix
    img_name = data['imagePath'][:-4]
    # get points of the polygon
    data = np.array(data['shapes'])
    # if label is femur, masking it with intensity 1.
    # if label is tibia, masking it with intensity 2.
    for point in range(data.shape[0]):
        if data[point]['label'] == 'femur':
            poly = data[point]['points']
            poly = np.array(poly)
            mask = cv.fillPoly(mask, np.int32([poly]), color=(0, 255, 0))
        elif data[point]['label'] == 'tibia':
            poly = data[point]['points']
            poly = np.array(poly)
            mask = cv.fillPoly(mask, np.int32([poly]), color=(0, 0, 255))

    # check whether mask is correct
    # cv.imshow('test', mask)
    # cv.waitKey(0)
    # mask = mask / 255.0
    # Put the directory path of the mask on save_dir
    cv.imwrite(save_dir + img_name + ".png", mask)
