import cv2 as cv
import os


def resize_with_pad(img, target_size, rgb=False):
    """resize the image by padding at borders.
    Params:
        img: image to be resized, read by cv2.imread()
        target_size: a tuple shows the image size after padding.
        For example, a tuple could be like (width, height)
    Returns:
        image: resized image with padding
    refer to
    https://stackoverflow.com/questions/43391205/add-padding-to-images-to-get-them-into-the-same-shape
    """
    img_size = (img.shape[1], img.shape[0])
    d_w = target_size[0] - img_size[0]
    d_h = target_size[1] - img_size[1]
    top, bottom = d_h // 2, d_h - (d_h // 2)
    left, right = d_w // 2, d_w - (d_w // 2)
    # rgb image default read by cv2, with BGR channels
    if rgb:
        pad_image = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=(255, 0, 0))
    else:
        pad_image = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT)
    return pad_image


def preprocessing(input_path, side="right"):
    '''
    preprocessing raw image for network training/inference
    :param input_path: The path of the raw images
    :param side: crop the image with right or left side
    :return: preprocessed image
    '''
    img_names = os.listdir(input_path)
    for img_name in img_names:
        # read image
        img_path = os.path.join(input_path, img_name)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        h, w = img.shape
        img = cv.resize(img, (256, int(h * 256 / w)))
        # cropped the image to half, with unilateral limb
        if side == "right":
            cropped_img = img[:, :128]
        else:
            cropped_img = img[:, 128:]
            cropped_img = cv.flip(cropped_img, 1)

        img = resize_with_pad(cropped_img, (128, 1024))
        # image enhancement
        clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        # img = cv.equalizeHist(img)
        # cv.imshow("1", img)
        # cv.waitKey(0)
        # img_out_path = os.path.join(output_path, img_name[:-4]+"_{}".format(side[0])+".jpg")
        img_out_path = os.path.join(output_path, img_name)
        cv.imwrite(img_out_path, img)


if __name__ == '__main__':
    input_path = "../data/raw/"
    output_path = "../data/preprocessed/images"
    preprocessing(input_path, side="right")
