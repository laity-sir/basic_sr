import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image

def image_h5(args):
    h5_file = h5py.File(args.output_path, 'w')
    hr_group = h5_file.create_group('hr')
    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr = np.array(hr).astype(np.float32)
        hr_group.create_dataset(str(i), data=hr)
    h5_file.close()
    print('over')
def read_h5(file,index):
    with h5py.File(file, 'r') as f:
        index=index%len(f['hr'])
        img=f['hr'][str(index)][:,:].astype(np.uint8)
        img=pil_image.fromarray(img)
        return img
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default='../test/Set5')
    parser.add_argument('--output-path', type=str, default='./hh.h5')
    args = parser.parse_args()
    image_h5(args)
