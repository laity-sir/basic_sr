import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils.utils import  convert_rgb_to_y
from PIL import Image
from torchvision import transforms
transform=transforms.Compose([
    transforms.ToTensor()
])

def train(args):
    h5_file = h5py.File(args.output_path, 'w')
    lr_patches = []
    hr_patches = []
    flip = [0,1]
    count = 1

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_images = []

        if args.with_aug:
            for s in [1.0]:
                for r in [90,180,270,0]:
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
            hr_width = (hr.width // args.scale) * args.scale
            hr_height = (hr.height // args.scale) * args.scale
            hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)  ##高分辨图片
            for i in range(0,hr_height-args.patch_size,args.patch_size):
                for j in range(0,hr_width-args.patch_size,args.patch_size):
                    # count+=1
                    hr1 =  hr.crop((i, j, i + args.patch_size, j + args.patch_size))
                    lr1 = hr1.resize((hr1.width // args.scale, hr1.height // args.scale), resample=pil_image.BICUBIC)
                    count += 1
                    # hr1.save(os.path.join('./data/hr/', '{}.png'.format(count)))
                    # lr1.save(os.path.join('./data/lr/','{}.png'.format(count)))
                    hr1 = np.array(hr1).astype(np.float32)
                    lr1 = np.array(lr1).astype(np.float32)
                    if args.mode=='y':
                        hr1 = convert_rgb_to_y(hr1)
                        lr1 = convert_rgb_to_y(lr1)
                    hr1=transform()(hr1)
                    lr1=transform()(lr1)
                    lr_patches.append(lr1)
                    hr_patches.append(hr1)
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    h5_file.close()
    print('over')
    print(count)


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        if args.mode=='y':
            hr = convert_rgb_to_y(hr)
            lr = convert_rgb_to_y(lr)
        hr=transform()(hr)
        lr=transform()(lr)
        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()
    print('over')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default='./data/T91')
    parser.add_argument('--output-path', type=str, default='./train/2/hr.h5')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--patch-size', type=int, default=48)
    parser.add_argument('--with-aug', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--mode', type=str,help='y:表示ycbcr')
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)
