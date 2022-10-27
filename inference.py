import torch
import PIL.Image as Image
from torchvision import transforms
import cv2 as cv
from utils.prediction2rgbmask import pred2mask
import os
import numpy as np

'''
predict the segmentation with pretrain model.
'''

img_path = "./data/inference/images"
save_path = "./data/inference/results/"
model_path = "exp_1_128x1024_dc_model.pth"


def inference(img_path, model, save_path=save_path):
    model.eval()
    image_names = os.listdir(img_path)
    masks = []
    imgs = []

    with torch.no_grad():
        for name in image_names:
            img_dir = os.path.join(img_path, name)
            img = Image.open(img_dir).convert("L")
            imgs.append(img)
            # img = transforms.Resize((256, 256))(img)
            img_tensor = transforms.ToTensor()(img)
            img_tensor = img_tensor.cuda().unsqueeze(0)
            pred = model(img_tensor)
            mask = pred2mask(pred)

            mask = np.array(mask) * 255
            mask = cv.cvtColor(mask, cv.COLOR_RGB2BGR)

            masks.append(mask)
            cv.imwrite(save_path + name[:-4] + '.png', mask)


if __name__ == '__main__':
    model = torch.load(model_path)
    inference(img_path, model)
