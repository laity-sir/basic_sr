import h5py
from torch.utils.data import Dataset
from torchvision import transforms
import sys
import numpy as np
sys.path.append('./model')
transform1=transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])
class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            idx=idx%len(f['lr'])
            lr=f['lr'][idx]
            hr=f['hr'][idx]
            if lr.ndim==2: ###说明是y通道
                lr=np.expand_dims(lr,2)
                hr=np.expand_dims(hr,2)
            ###数据增强

            lr=transforms.ToTensor()(lr)
            hr=transforms.ToTensor()(hr)
            return lr,hr
    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

class EvalDataset(Dataset):
    def __init__(self, h5_file,mode='y'):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file
        self.mode=mode
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr=f['lr'][str(idx)][:,:]
            hr=f['hr'][str(idx)][:, :]
            if lr.ndim==2:
                lr=np.expand_dims(lr,2)
                hr=np.expand_dims(hr,2)
            lr=transforms.ToTensor()(lr)
            hr=transforms.ToTensor()(hr)
            return lr,hr
    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import random
    import numpy as np
    num=72
    col=8
    row=int(num/8)
    dataset=TrainDataset('./t91.h5')
    print(dataset[0][1])
    index = np.random.randint(1, len(dataset), num)
    print(index)
    hh=[dataset[i][0] for i in index]
    ###可视化图片
    for i in range(num):
        for j in range(8):
            # plt.figure()
            plt.subplot(row, col, i+1)
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(transforms.ToPILImage()(hh[i]), cmap='gray')
    plt.show()

