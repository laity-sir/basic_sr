import math
import random
from typing import Any
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt


def rgb2ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    """Implementation of rgb2ycbcr function in Matlab under Python language

    Args:
        image (np.ndarray): Image input in RGB format.
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data

    """
    if only_use_y_channel:
        image = np.dot(image, [65.481, 128.553, 24.966]) + 16.0
    else:
        image = np.matmul(image, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]])/255 + [
            16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image

def show_tensor_img(tensor_img):
    to_pil = transforms.ToPILImage()
    img1 = tensor_img.cpu().clone()
    if img1.dim()==4:
        img1=img1.squeeze(0)
    img1 = to_pil(img1)
    plt.imshow(img1,cmap='gray')



if __name__=='__main__':
    img=cv2.imread('./data/test/Set5/baby.png')
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img)
    img=rgb2ycbcr(img,only_use_y_channel=False)
    print(img)
    print(img.shape)
    # img1=Image.fromarray(img)
    # img1.show()
    cv2.imshow('img',img)
    cv2.waitKey(0)
