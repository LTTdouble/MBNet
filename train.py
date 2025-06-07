import os

import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchnet import meter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import time

import model_ours
from eval import PSNR, SSIM, SAM
from option import opt

from data_utils import TrainsetFromFolder, ValsetFromFolder
import torch

from scipy.stats import pearsonr
from torch.optim.lr_scheduler import MultiStepLR

import torch.nn.init as init
import scipy.io as scio
psnr = []
out_path = './result/' + opt.datasetName + '/'

log_interval = 50
per_epoch_iteration = 10
total_iteration = per_epoch_iteration*opt.end_epoch
class FeatureExtractor(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def main():
    # if opt.show:
    global best_psnr
    global writer
    if opt.cuda:
        # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    # Loading datasets

    train_set = TrainsetFromFolder('./dataset/trains/' + opt.datasetName + '/' + str(opt.upscale_factor) + '/')
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    val_set = ValsetFromFolder('./dataset/evals/' + opt.datasetName + '/' + str(opt.upscale_factor) + '/')
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    #
    model = model_ours.MBNet(nf=opt.n_feats)

    criterion = nn.L1Loss()

    # criterion =loss.HybridLoss_1(weight=opt.alpha)

    print('Parameters number is ', sum(param.numel() for param in model.parameters()))


    if opt.cuda:

        model = model.cuda()
        criterion = criterion.cuda()

    else:
        model = model.cpu()
    print('# parameters:', sum(param.numel() for param in model.parameters()))

    # Setting Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-08)

    # optionally resuming from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

            # Setting learning rate
    # scheduler = MultiStepLR(optimizer, milestones=[35, 70, 105, 140, 175], gamma=0.5, last_epoch=-1)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60,80,100], gamma=0.5, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,opt.nEpochs,eta_min=2e-5)

    writer = SummaryWriter(log_dir='logs/' + opt.datasetName + opt.method+ '_' + str(time.ctime()))
    # Training
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
        train(train_loader, optimizer, model, criterion, epoch)
        # train(train_loader, optimizer, model, criterion, epoch, FDL_loss)
        val(val_loader, model, epoch)
        save_checkpoint(epoch, model, optimizer)
        scheduler.step()

    scio.savemat(out_path + 'LFF1GRL0.mat', {'psnr': psnr})  # , 'ssim':ssim, 'sam':sam})


def pad_and_group_channels(image):
    group_size = 7

    B, N, H, W = image.shape

    remainder = N % group_size

    if remainder != 0:
        pad_size = group_size - remainder
        pad = image[:, -pad_size - 1:-1, :]
        image = torch.cat([image, pad],1)

    else:
        image =image

    return image


if opt.upscale_factor == 3:

        nearest_LSR = nn.Upsample(scale_factor=1/(opt.upscale_factor), mode='bicubic')

else:

        nearest_LSR = nn.Upsample(scale_factor=1/(opt.upscale_factor//2), mode='bicubic')

def calculate_corr_matrix(input):
    corr_matrix = torch.zeros(input.size(1), input.size(1))
    for i in range(input.size(1)):
        for j in range(input.size(1)):
            corr, _ = pearsonr(input[:, i].numpy().flatten(), input[:, j].numpy().flatten())
            corr = np.clip(corr, -1.0, 1.0)
            corr_matrix[i, j] = corr
    return corr_matrix


def group_channels(input, threshold, corr_matrix):
    groups = []
    visited = set()
    for i in range(input.size(1)):
        if i not in visited:
            group = [i]
            visited.add(i)
            for j in range(i + 1, input.size(1)):
                if j not in visited and corr_matrix[i, j] > threshold:
                    group.append(j)
                    visited.add(j)
            groups.append(group)
    return groups

nearest = nn.Upsample(scale_factor=opt.upscale_factor, mode='nearest')

def train(train_loader, optimizer, model, criterion, epoch):
    for iteration, batch in enumerate(train_loader, 1):
        input, label = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        group_size = 7
        num_channels = input.shape[1]
        split_points = range(0, num_channels, group_size)
        splitted_images = np.array_split(input, split_points, axis=1)
        input1 = pad_and_group_channels(input)


        if opt.cuda:

            input = input.cuda()
            input1 = input1.cuda()
            label = label.cuda()

        for i in range(2):
            SR, frist_result = model(input, input1, splitted_images)

            label_LR = nearest_LSR(label).cuda()

            loss = criterion(SR, label)+criterion(frist_result,label_LR)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # epoch = epoch + 1
            writer.add_scalar('scalar/train_loss', loss, epoch)
    if (iteration + log_interval) % log_interval == 0:
        print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(train_loader), loss.item()))

def val(val_loader, model, epoch):
    model.eval()
    epoch_meter = meter.AverageValueMeter()
    epoch_meter.reset()
    model.eval()
    val_psnr = 0
    val_ssim = 0
    val_sam = 0

    with torch.no_grad():
        for iteration, batch in enumerate(val_loader, 1):
            input, HR = Variable(batch[0], volatile=True), Variable(batch[1])
            input1 = pad_and_group_channels(input)
            group_size= 7

            num_channels = input1.shape[1]
            split_points = range(0, num_channels, group_size)
            splitted_images = np.array_split(input1, split_points, axis=1)

            if opt.cuda:

                input1 = input1.cuda()
                input = input.cuda()

            SR, frist_result = model(input, input1, splitted_images)
            SR= SR.cpu().data[0].numpy().astype(np.float32)

            val_psnr += PSNR(SR, HR.data[0].numpy())
            val_ssim += SSIM(SR, HR.data[0].numpy())
            val_sam += SAM(SR, HR.data[0].numpy())

        print("PSNR = {:.3f}   SSIM = {:.4F}    SAM = {:.3f}".format(val_psnr / len(val_loader), val_ssim / len(val_loader),
                                                                 val_sam / len(val_loader)))

        val_psnr = val_psnr / len(val_loader)
    psnr.append(val_psnr)
    writer.add_scalar('Val/PSNR', val_psnr, epoch)

def save_checkpoint(epoch, model, optimizer):
    model_out_path = "checkpoint/" + "{}_model_{}_epoch_{}.pth".format(opt.datasetName, opt.upscale_factor, epoch)
    state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")
    torch.save(state, model_out_path)


if __name__ == "__main__":
    main()