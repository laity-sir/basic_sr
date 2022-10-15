import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from utils.utils import convert_rgb_to_y,plot_data_loader_image,seed_torch,train_hr_transform,test_hr_transform,train_lr_transform,test_lr_transform
import h5py
from image_h5 import read_h5
"""
从h5文件读取数据
"""
class dataset(Dataset):
    def __init__(self, path, scale,patch_size=96,mode='',train=True,num=10):
        """
        path:高分辨率图片的路径
        scale:尺度因子
        patch_size=图片的大小
        mode:如果mode='y',则表示输入图片是y通道，否则是rgb格式
        train:如果为True,则表示训练集。
        """
        super(dataset, self).__init__()
        self.mode=mode
        self.train=train
        self.num=num
        self.scale = scale
        self.patch_size=patch_size
        self.file=path
    def __len__(self):
        with h5py.File(self.file, 'r') as f:
            if self.train:
                return len(f['hr'])*self.num
            else:
                return len(f['hr'])

    def __getitem__(self, index):
        hr=read_h5(self.file,index)
        if self.mode=='y':
            y=convert_rgb_to_y(np.array(hr)).astype(np.uint8)
            y=Image.fromarray(y)
            hr=y
        else:
            hr=hr
        if True:
            hr_width = (hr.width // self.scale) * self.scale
            hr_height = (hr.height // self.scale) * self.scale
            hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
        ###########################################################
        if self.train:
            hr=train_hr_transform(crop_size=self.patch_size)(hr)
            lr=train_lr_transform(crop_size=self.patch_size,scale=self.scale)(hr)
        else:
            hr=test_hr_transform()(hr)
            _,h,w=hr.shape
            lr=test_lr_transform(w,h,self.scale)(hr)
        return lr,hr

if __name__=='__main__':
    seed_torch()
    dataset1=dataset(path='./hh.h5', scale=2, patch_size=48, train=1, num=16)
    dataloader=torch.utils.data.DataLoader(dataset=dataset1,
                                           batch_size=8,
                                           shuffle=True,
                                           num_workers=0
                                           )
    plot_data_loader_image(data_loader=dataloader)
