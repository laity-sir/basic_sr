import torch
import numpy as np
from torchvision import transforms
import random
import cv2
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('./model')
from PIL import Image

def calc_patch_size(func):
    def wrapper(args):
        if args.scale == 2:
            args.patch_size = 48
        elif args.scale == 3:
            args.patch_size = 48
        elif args.scale == 4:
            args.patch_size = 48
        else:
            raise Exception('Scale Error', args.scale)
        return func(args)
    return wrapper

def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def convert_rgb_to_ycbcr(img, dim_order='hwc'):
    if dim_order == 'hwc':
        y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
        cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
        cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    else:
        y = 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
        cb = 128. + (-37.945 * img[0] - 74.494 * img[1] + 112.439 * img[2]) / 256.
        cr = 128. + (112.439 * img[0] - 94.154 * img[1] - 18.285 * img[2]) / 256.

    return np.array([y, cb, cr]).transpose([1, 2, 0])  ###chw  hwc  输出的array  数据类型是float

def convert_ycbcr_to_rgb(img, dim_order='hwc'):
    if dim_order == 'hwc':
        r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
        g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
        b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
    else:
        r = 298.082 * img[0] / 256. + 408.583 * img[2] / 256. - 222.921
        g = 298.082 * img[0] / 256. - 100.291 * img[1] / 256. - 208.120 * img[2] / 256. + 135.576
        b = 298.082 * img[0] / 256. + 516.412 * img[1] / 256. - 276.836
    return np.array([r, g, b]).transpose([1, 2, 0])

'''
# =======================================
# image processing process on numpy image
# channel_convert(in_c, tar_type, img_list):
# rgb2ycbcr(img, only_y=True):
# bgr2ycbcr(img, only_y=True):
# ycbcr2rgb(img):
# modcrop(img_in, scale):
# =======================================
'''
def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:  ####这里是说如果输入的不是uint，那么就是floa，需要乘以255
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def shave(img_in, border=0):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    h, w = img.shape[:2]
    img = img[border:h - border, border:w - border]
    return img

def preprocess_to_y(img, device='cpu'):
    """
    将图片转换为 y通道的数据
    """
    img = np.array(img).astype(np.float32)   ##将PIL图片转换为numpy  数据类型是np.float32
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr  ####这里是为了产生测试图片  所以会转换为tensor格式
def preprocess_to_tesnor(img, device):
    """
    将图片转换为 tensor数据
    img:传入的数据可以是pil和numpy格式
    """
    img=transforms.ToTensor()(img).to(device)
    if img.dim()==3:
        img=img.unsqueeze(0)
    return img
def preprocess_to_pil(tensor):
    if tensor.dim()>3:
        tensor=tensor.squeeze(0)
    img=transforms.ToPILImage()(tensor)

    return img
def get_hr_lr_bicubic(path,scale):
    image = Image.open(path).convert('RGB')
    image_width = (image.width // scale) * scale
    image_height = (image.height // scale) * scale
    hr = image.resize((image_width, image_height), resample=Image.BICUBIC)

    lr = hr.resize((hr.width // scale, hr.height // scale), resample=Image.BICUBIC)
    bicubic = lr.resize((lr.width *scale, lr.height *scale), resample=Image.BICUBIC)
    return hr,lr,bicubic

def merge(y,ycbcr):
    """
    主要是将y通道数据和cbcr拼接起来
    y:
    ycbcr:是通过convert_ycbcr_to_rgb得到的，格式是array
    """
    if torch.is_tensor(y):
        y=y.squeeze(0)
        y=transforms.ToPILImage()(y)
        y=np.array(y)
    output = np.array([y, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])  ###hwc   chw
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = Image.fromarray(output)
    return output

def calc_psnr(img1, img2):
    # imdff = torch.clamp(img1, 0, 1) - torch.clamp(img2, 0, 1)
    """
    img1:
    img2:都是tensor格式
    """
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def random_vertically_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """随机翻转
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

def random_crop(image, crop_shape):
    """
    随机裁剪图片
    注意:这里是用pil进行裁剪的,所以image必须是pil
    crop_shape:必须是数组
    """
    ###image 是pil读取的，crop_shape是裁剪的大小
    nw = random.randint(0, image.size[0] - crop_shape[0])  ##裁剪图像在原图像中的坐标
    nh = random.randint(0, image.size[1] - crop_shape[1])
    image_crop = image.crop((nh, nw, nh + crop_shape[0], nw + crop_shape[1]))

    return image_crop
def center_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    """Crop small image patches from one image center area.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        image_size (int): The size of the captured image area.

    Returns:
        patch_image (np.ndarray): Small patch image

    """
    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = (image_height - image_size) // 2
    left = (image_width - image_size) // 2

    # Crop image patch
    patch_image = image[top:top + image_size, left:left + image_size, ...]

    return patch_image


def random_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    """Crop small image patches from one image. 随机中心旋转

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        image_size (int): The size of the captured image area.

    Returns:
        patch_image (np.ndarray): Small patch image

    """
    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = random.randint(0, image_height - image_size)
    left = random.randint(0, image_width - image_size)

    # Crop image patch
    patch_image = image[top:top + image_size, left:left + image_size, ...]

    return patch_image


def random_rotate(image,
                  angles: list,
                  center = None,
                  scale_factor = 1.0) -> np.ndarray:
    """Rotate an image by a random angle  随机旋转

    Args:
        image (np.ndarray): Image read with OpenCV
        angles (list): Rotation angle range
        center (optional, tuple[int, int]): High resolution image selection center point. Default: ``None``
        scale_factor (optional, float): scaling factor. Default: 1.0

    Returns:
        rotated_image (np.ndarray): image after rotation

    """
    image_height, image_width = image.shape[:2]

    if center is None:
        center = (image_width // 2, image_height // 2)

    # Random select specific angle
    angle = random.choice(angles)
    matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
    rotated_image = cv2.warpAffine(image, matrix, (image_width, image_height))

    return rotated_image


def random_horizontally_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip the image upside down randomly 随机翻转

    Args:
        image (np.ndarray): Image read with OpenCV
        p (optional, float): Horizontally flip probability. Default: 0.5

    Returns:
        horizontally_flip_image (np.ndarray): image after horizontally flip

    """
    if random.random() < p:
        horizontally_flip_image = cv2.flip(image, 1)
    else:
        horizontally_flip_image = image

    return horizontally_flip_image



def psnr(img1, img2,border=0):
    """
    function:计算两幅图片的psnr
    img1:格式可以是tensor,也可以是array,pil
    """
    ##这里通常会去除边框
    if torch.is_tensor(img1):
        if img1.dim() > 3:
            #大于3,说明包含了batch维度,需要去除
            img1 = img1.squeeze(0)
        img1 = transforms.ToPILImage()(img1)
    if torch.is_tensor(img2):
        if img2.dim() > 3:
            img2 = img2.squeeze(0)
        img2 = transforms.ToPILImage()(img2)
    img1=np.array(img1)
    img2=np.array(img2)
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))
    # imdff = np.array(img2).astype(np.float64) - np.array(img1).astype(np.float64)
    # rmse = np.sqrt(np.mean(imdff**2))
    # ps = 20*np.log10(255/rmse)
    # return ps


def calculate_ssim(img1, img2):
    """
    计算ssim
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def ssim(img1, img2,border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    输入的参数可以是numpy格式，也可以是tensor格式
    '''
    if torch.is_tensor(img1):
        if img1.dim()>3:
            img1=img1.squeeze(0)
        img1=transforms.ToPILImage()(img1)
    if torch.is_tensor(img2):
        if img2.dim()>3:
            img2=img2.squeeze(0)
        img2=transforms.ToPILImage()(img2)
    img1=np.array(img1).astype(np.float64)
    img2=np.array(img2).astype(np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]
    if img1.ndim == 2:
        return calculate_ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(calculate_ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return calculate_ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def show_tensor_img(tensor_img):
    to_pil = transforms.ToPILImage()
    img1 = tensor_img.cpu().clone()
    if img1.dim()==4:
        img1=img1.squeeze(0)
    img1 = to_pil(img1)
    plt.imshow(img1,cmap='gray')

def plot_data_loader_image(data_loader):
    """
    显示dataloader中所有图片
    col:表示一行显示多少图片
    """
    batch_size = data_loader.batch_size
    col = batch_size
    row=int(batch_size*len(data_loader)/col)
    for i,data in enumerate(data_loader):
        lr, hr = data   ###  data是数据
        img=hr
        for j in range(col):
            plt.subplot(row, col, i*col+j+1)
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            # plt.title('{}batch {} figure'.format(i,i*col+j+1))
            show_tensor_img(img[j])
            if (i * col + j+1) >=(batch_size * len(data_loader)):
                break
    plt.show()


def seed_torch(seed=1218):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def save_checkpoint(path,model, epoch, train_loss, test_loss, optimizer,test_psnr,test_ssim,best_epoch=0):
    model_out_path = os.path.join(path , "model.pth")  ###保存的地址
    state = {"epoch": epoch,
             "model": model,
             'train_loss': train_loss,
             'test_loss': test_loss,
             'optimizer': optimizer.state_dict(),
             'test_psnr':test_psnr,
             'test_ssim':test_ssim,
             'best_epoch':best_epoch
             }
    # check path status
    if not os.path.exists("../model/"):
        os.makedirs("../model/")
    # save model
    torch.save(state, model_out_path)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif','bmp'])

def train_lr_transform(crop_size,scale):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(crop_size//scale,interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

def train_hr_transform(crop_size):
    return transforms.Compose([
        # transforms.CenterCrop(crop_size),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ])

def test_hr_transform():
    return transforms.Compose([
        transforms.ToTensor()  ###hr.shape[2],hr.shape[1]
    ])
def test_lr_transform(w,h,scale):
    w=int(w/scale)
    h=int(h/scale)
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((h,w),interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

