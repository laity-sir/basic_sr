import math
import random
from typing import Any
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

def random_vertically_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip an image horizontally randomly

    Args:
        image (np.ndarray): Image read with OpenCV
        p (optional, float): Vertically flip probability. Default: 0.5

    Returns:
        vertically_flip_image (np.ndarray): image after vertically flip

    """
    if random.random() < p:
        vertically_flip_image = cv2.flip(image, 0)
    else:
        vertically_flip_image = image

    return vertically_flip_image

if __name__=='__main__':
    image=Image.open('./data/test/Set5/baby.png')
    image.show(image)
    image=np.array(image)
    image=random_vertically_flip(image)
    hh=Image.fromarray(image)
    hh.show()
    # image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    # plt.imshow(image)
    # plt.show()