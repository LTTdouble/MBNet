''' network architecture for MBNet '''

import torch
import torch.nn as nn
from option import opt
import BFFN_arch
import common_srdnet
import Mamba_arch

class MBNet(nn.Module):
    def __init__(self, nf=64):
        super(MBNet, self).__init__()
        self.nf = nf

        kernel_size = 3
        n_layers = 1
        hidden_dim = []
        for i in range(n_layers):
            hidden_dim.append(nf)

        conv2d = []
        conv2d.append((nn.Conv2d(nf, nf, kernel_size=( 1, 1), stride=1, padding=( 0, 0))))
        self.conv2d = nn.Sequential(*conv2d)

        #### activation function
        self.act = nn.ReLU(True)

        scale_l = 2
        tail_l = []

        if opt.upscale_factor == 3:
            tail_l.append(nn.ConvTranspose2d(nf,nf, kernel_size=(3, 3),
                                             stride=(1, 1), padding=(1, 1), bias=False))

            self.nearest_l = nn.Upsample(scale_factor=1, mode='bicubic')

        else:
            tail_l.append(nn.ConvTranspose2d(nf, nf, kernel_size=(2 + scale_l, 2 + scale_l),
                                             stride=(scale_l, scale_l), padding=(1, 1), bias=False))

            self.nearest_l = nn.Upsample(scale_factor=scale_l, mode='bicubic')

        self.tail_l = nn.Sequential(*tail_l)

        scale_g = opt.upscale_factor//2
        tail_g = []

        if opt.upscale_factor == 3:
            tail_g.append(nn.ConvTranspose2d(nf, nf, kernel_size=(2 + opt.upscale_factor, 2 + opt.upscale_factor),
                                             stride=(opt.upscale_factor, opt.upscale_factor), padding=(1, 1), bias=False))

        else:
            tail_g.append(nn.ConvTranspose2d(nf, nf, kernel_size=(2 + scale_g, 2 + scale_g),
                                             stride=(scale_g, scale_g), padding=(1, 1), bias=False))

        self.tail_g = nn.Sequential(*tail_g)

        self.bffn= BFFN_arch.BFFN(n_feats = opt.n_feats)

        self.nearest = nn.Upsample(scale_factor=opt.upscale_factor, mode='bicubic')
        self.head_l = nn.Conv2d(opt.n_colors, nf, kernel_size=3, stride=1, padding=(1, 1))

        conv = common_srdnet.default_conv

        RBs = [
            common_srdnet.ResBlock(
                conv, nf, kernel_size, act=self.act, res_scale=0.1
            ) for _ in range(4)
        ]

        self.RB = nn.Sequential(*RBs)

        self.Mamba_Net = Mamba_arch.Mamba(in_chans=nf,embed_dim=nf,depths=[1, 1],d_state=16)  #  The number of depths and stages can be adjusted to balance performance and computational complexity.

        self.out_end = nn.Conv2d(nf, 7, kernel_size=3, stride=1, padding=1)

        self.head_group_head = nn.Conv2d(7, nf, kernel_size=3, stride=1, padding=1)

        self.gamma = nn.Parameter(torch.ones(3))
        self.gamma_rnn = nn.Parameter(torch.ones(2))

        self.SR_frist = nn.Conv2d(opt.n_colors, nf, kernel_size=3, stride=1, padding=1)

        self.SR_end = nn.Conv2d(nf, opt.n_colors, kernel_size=3, stride=1, padding=1)

        self.rnn_reduce = nn.Conv2d(2*nf, nf, kernel_size=1, stride=1, padding=0)

        self.reduceD = nn.Conv2d(nf * 3, nf, kernel_size=(1, 1), stride=1, padding=(0, 0))

    def forward(self, input, input1, splitted_images):
        bicu = self.nearest(input)

        group_size = 7
        LSR = []

        LRx = None

        for j in range(len(splitted_images) - 1):
            input2 = input1[:, j * group_size:(j + 1) * group_size, :, :]

            start_idx = -((j + 1) * group_size)
            end_idx = -(j * group_size) if j != 0 else None
            input3 = input1[:, start_idx:end_idx, :, :]

            bicu_single = self.nearest_l(input2)

            xL = self.RB(self.head_group_head(input2))

            L1_fea=self.head_group_head(input2)

            L2_fea =self.act(self.head_group_head(input3))

            L_fea = self.act(self.rnn_reduce(torch.cat([L1_fea, L2_fea], 1)))

            H = []

            for i in range(3):
                lstm_feats = self.Mamba_Net(L_fea)
                H.append(lstm_feats * self.gamma[i])

            lstm_feats_Mamba = torch.cat(H, 1)

            lstm_feats_Mamba = self.reduceD(lstm_feats_Mamba)

            lstm_feats=self.RB( lstm_feats_Mamba)+lstm_feats_Mamba

            frist_up=self.tail_l(lstm_feats)

            if j != 0:
                frist_up = torch.cat([self.gamma_rnn[0] * frist_up, self.gamma_rnn[1] * LRx], 1)
                frist_up = self.rnn_reduce(frist_up)

            frist_up = frist_up + self.nearest_l(xL)

            LRx = frist_up

            frist_up = self.out_end(frist_up) + bicu_single

            LSR.append(frist_up)

        frist_result = torch.cat(LSR, 1)
        if input.shape[1] % group_size != 0:
            frist_result = frist_result[:, :-(group_size - input.shape[1] % group_size), :, :]

        end2 = self.bffn(self.SR_frist(frist_result))
        end2 =self.RB(end2)+end2

        out2 = self.tail_g(end2)

        SR = self.SR_end(out2) + bicu

        return  SR,frist_result



