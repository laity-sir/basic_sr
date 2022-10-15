import argparse
import os
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import  convert_rgb_to_y,random_crop
from PIL import Image

def train(args):
    h5_file = h5py.File(args.output_path, 'w')
    lr_patches = []
    hr_patches = []
    flip = [0,1,2]
    count = 0
    if args.mode=='y':
        print('生成y通道数据')
    for image_path in sorted(os.listdir(args.images_dir)):
        hr = pil_image.open(os.path.join(args.images_dir,image_path)).convert('RGB')
        hr_images = []
        if args.mode=='y':
            hr=np.array(hr)
            hr = convert_rgb_to_y(hr).astype(np.uint8)
            hr=Image.fromarray(hr)
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        if args.with_aug:
            for s in [1.0,0.7,0.5]:
                for r in [0,90,180,270]:
                    for c in range(len(flip)):
                        tmp = hr.resize((int(hr.width * s), int(hr.height * s)), resample=pil_image.BICUBIC)
                        if flip[c] == 1:  ##水平翻转
                            tmp = tmp.transpose(Image.FLIP_LEFT_RIGHT)
                        elif flip[c] == 2:  ###垂直翻转
                            tmp = tmp.transpose(Image.FLIP_TOP_BOTTOM)
                        else:
                            tmp = tmp  ##不翻转
                        tmp = tmp.rotate(r, expand=True)
                        hr_images.append(tmp)
        else:
            hr_images.append(hr)

        for hr in hr_images:
            if args.samble_method=='m':
                for i in range(0,hr_height-args.patch_size,args.patch_size):
                    for j in range(0,hr_width-args.patch_size,args.patch_size):
                        hr1 =  hr.crop((i, j, i + args.patch_size, j + args.patch_size))
                        lr1 = hr1.resize((hr1.width // args.scale, hr1.height // args.scale),
                                         resample=pil_image.BICUBIC)
                        count += 1
                        # hr1.save(os.path.join('./data/hr/', '{}.png'.format(count)))
                        # lr1.save(os.path.join('./data/lr/','{}.png'.format(count)))
                        hr1 = np.array(hr1)
                        lr1 = np.array(lr1)
                        lr_patches.append(lr1)
                        hr_patches.append(hr1)

            else:
                for i in range(args.samble_num):
                    hr1=random_crop(hr,(args.patch_size,args.patch_size))
                    lr1 = hr1.resize((hr1.width // args.scale, hr1.height // args.scale), resample=pil_image.BICUBIC)
                    count+=1
                    # hr1.save(os.path.join('./data/hr/', '{}.png'.format(count)))
                    # lr1.save(os.path.join('./data/lr/','{}.png'.format(count)))
                    hr1 = np.array(hr1)
                    lr1 = np.array(lr1)
                    lr_patches.append(lr1)
                    hr_patches.append(hr1)
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    h5_file.close()
    print('over,训练集数量{}'.format(count))



def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    count=0
    if args.mode=='y':
        print('生成y通道数据')
    for i, image_path in enumerate(sorted(os.listdir(args.images_dir))):
        hr = pil_image.open(os.path.join(args.images_dir,image_path)).convert('RGB')
        if args.mode=='y':  ###这里是生成y通道数据
            hr=np.array(hr)
            hr = convert_rgb_to_y(hr).astype(np.uint8)
            hr=Image.fromarray(hr)
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr)
        lr = np.array(lr)
        count+=1
        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()
    print('测试集数量：{}'.format(count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default='./data/test/Set5')
    parser.add_argument('--output-path', type=str, default='./set5.h5')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--samble-method', type=str, default='',help='m 表示全部采样，否则就是随机采样')
    parser.add_argument('--samble-num', type=int, default=16,help='随机采样的个数')
    parser.add_argument('--patch-size', type=int, default=24)
    parser.add_argument('--with-aug',type=bool,default=True,help='数据增强，8倍')
    parser.add_argument('--eval', type=bool,default='1')
    parser.add_argument('--mode', type=str,help='y:表示ycbcr',default='y')
    args = parser.parse_args()
    if not args.eval:
        train(args)
    else:
        eval(args)
