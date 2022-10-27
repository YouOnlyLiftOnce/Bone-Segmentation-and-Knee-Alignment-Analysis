import cv2 as cv
import os
from utils.preprocessing import resize_with_pad
from tqdm import tqdm
import albumentations as A

'''
Performing a offline data augmentation, including rotation, scaling, and translation.
The image and mask is from /data/1_preprocessed, and result is writen in /data/1_preprocessed/augmentation.
After augmentation, augmented images and masks are manually copy and paste to data/train, and data/test folds
'''

img_dir = "../data/1_preprocessed/images/"
mask_dir = "../data/1_preprocessed/masks/"

img_save_dir = "../data/1_preprocessed/augmentation/images/"
mask_save_dir = "../data/1_preprocessed/augmentation/masks/"

# transform applied both on images and masks
transform_common = A.Compose([
    A.Rotate((-30, 30), border_mode=cv.BORDER_CONSTANT, value=0, mask_value=(255, 0, 0), crop_border=False, p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, mode=cv.BORDER_CONSTANT, cval=0, cval_mask=(255, 0, 0))
])

for img_name in tqdm(os.listdir(img_dir)):
    if (img_name.endswith(".jpg")):
        for i in range(20):
            img = cv.imread(img_dir + img_name)
            img = resize_with_pad(img, (1024, 1024), rgb=False)

            mask_name = img_name[:-4] + ".png"
            mask = cv.imread(mask_dir + mask_name)
            mask = resize_with_pad(mask, (1024, 1024), rgb=True)
            # change postfix of mask to .jpg, since Augmentor requires image name and mask name to be identical

            transformed = transform_common(image=img, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']

            cv.imwrite(img_save_dir + f"{i}_" + img_name, transformed_image)
            cv.imwrite(mask_save_dir + f"{i}_" + mask_name, transformed_mask)
