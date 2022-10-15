import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
sys.path.append('model')
import copy
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model.msrn import MSRN
from datasets import TrainDataset,EvalDataset
from utils import AverageMeter,psnr,ssim,seed_torch,save_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='./t91.h5')
    parser.add_argument('--eval-file', type=str, default='./set5.h5')
    parser.add_argument('--outputs-dir', type=str, default='./network')
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--clip-grad', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--step", type=int, default=20,help="Sets the learning rate to the initial LR decayed by momentum every n epochs")
    parser.add_argument('--batch-size', type=int, default=16)  #
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
    parser.add_argument("--weight-decay", "--wd", default=1e-8, type=float, help="weight decay, Default: 1e-4")
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint, Default=None")
    parser.add_argument("--pretrained", default="", type=str, help='path to pretrained model, Default=None')
    parser.add_argument("--start-epoch", default=0, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--mode', type=str, default='y')
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    seed_torch()
    if args.mode=='y':
        print('数据是y通道的')
        model = MSRN(scale=args.scale, in_channels=1)
    else:
        print('数据是rgb格式的')
        model = MSRN(scale=args.scale,in_channels=3)
    if torch.cuda.device_count() > 1:
        print("使用多个gpu", torch.cuda.device_count(), 'gpus')
        model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    save_data={

    }
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
            optimizer.load_state_dict(checkpoint['optimizer'])
            save_data=torch.load('./fig/save_data.pth')
            print("===> loading checkpoint: {},start_epoch: {} ".format(args.resume,args.start_epoch))
        else:
            print("===> no checkpoint found at {}".format(args.resume))
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("===> load model {}".format(args.pretrained))
            weights = torch.load(args.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("===> no model found at {}".format(args.pretrained))

    train_dataset = TrainDataset(args.train_file)
    lenth=10
    train_dataset,_=torch.utils.data.random_split(train_dataset,[lenth,len(train_dataset)-lenth])
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    ###测试集
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.start_epoch, args.num_epochs+1):
        # 学习率衰减
        lr = args.lr * (0.5 ** ((epoch + 1) // args.step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        model.train()
        train_loss = AverageMeter()
        test_loss = AverageMeter()
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs))
            for index,data in enumerate(train_dataloader):
                if (epoch*len(train_dataloader)+index)%200000:
                    """
                    可以用來調節學習率
                    """
                    pass
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)
                loss = criterion(preds, labels)
                train_loss.update(loss.item(), len(inputs))
                optimizer.zero_grad()
                loss.backward()
#               #梯度裁剪
                nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.clip_grad / lr)
                optimizer.step()
                t.set_postfix(trainloss='{:.6f}'.format(train_loss.avg))
                t.update(len(inputs))
                del inputs, labels, loss
                torch.cuda.empty_cache()
        model.eval()
        test_psnr = AverageMeter()
        test_ssim = AverageMeter()
        for data in eval_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)
                loss = criterion(preds, labels)
                test_loss.update(loss.item(), len(inputs))
            test_psnr.update(psnr(preds, labels), len(inputs))
            test_ssim.update(ssim(preds, labels), len(inputs))
            ###释放不用的变量
            del inputs, labels, loss
            torch.cuda.empty_cache()
        print('eval psnr: {:.2f},eval ssim:{:.2f} eval loss :{:.2f}'.format(test_psnr.avg, test_ssim.avg, test_loss.avg))
        save_data[epoch]={'train_loss':train_loss.avg,'test_loss':test_loss.avg,'test_psnr':test_psnr.avg,'test_ssim':test_ssim.avg}
        torch.save(save_data,'./fig/save_data.pth')
        ###这里只保存最后一次的数据
        if test_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = test_psnr.avg
            best_weights = copy.deepcopy(model)
            print('best_epoch:',best_epoch)
            ###保存整个网络
            torch.save(best_weights, os.path.join('./network', 'best{}.pth'.format(args.scale)))
        save_checkpoint(args.outputs_dir, model, epoch, train_loss.avg, test_loss.avg, optimizer, test_psnr.avg, test_ssim.avg,best_epoch)
    print('best_epoch',best_epoch)