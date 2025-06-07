

import torch
import torch.nn as nn
import torch.nn.functional as F


class Channel_Atten(nn.Module):
    def __init__(self, in_channels=512, gamma_init=0.0, reduction=16, patch_size=4):

        super(Channel_Atten, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.reduction = reduction
        self.reduced_channels = max(4, in_channels // reduction)

        self.patch_conv = nn.Conv2d(in_channels, in_channels,
                                    kernel_size=patch_size,
                                    stride=patch_size,
                                    groups=in_channels)

        self.query = nn.Linear(in_channels, self.reduced_channels, bias=False)
        self.key = nn.Linear(in_channels, self.reduced_channels, bias=False)

        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def forward(self, x):
        batch_size, c, h, w = x.size()
        x_patch = self.patch_conv(x)
        x_patch = x_patch.view(batch_size, c, -1)

        query = self.query(x_patch.transpose(1, 2))
        key = self.key(x_patch.transpose(1, 2))

        energy = torch.bmm(query, key.transpose(1, 2))
        attention = F.softmax(energy, dim=-1)

        out = torch.bmm(attention, x_patch.transpose(1, 2))
        out = out.transpose(1, 2).view(batch_size, c, h // self.patch_size, w // self.patch_size)
        out = F.interpolate(out, size=(h, w), mode='nearest')

        return self.gamma * out


class Spatial_Atten(nn.Module):
    def __init__(self, in_channels=512, gamma_init=0.0, reduction=16, patch_size=4):

        super(Spatial_Atten, self).__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.reduction = reduction
        self.reduced_channels = max(4, in_channels // reduction)
        self.patch_conv = nn.Conv2d(in_channels, in_channels,
                                    kernel_size=patch_size,
                                    stride=patch_size,
                                    groups=in_channels)

        self.query = nn.Conv1d(in_channels, self.reduced_channels, 1)
        self.key = nn.Conv1d(in_channels, self.reduced_channels, 1)
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def forward(self, x):
        batch_size, c, h, w = x.size()

        x_patch = self.patch_conv(x)
        x_patch = x_patch.view(batch_size, c, -1)

        query = self.query(x_patch)
        key = self.key(x_patch)
        energy = torch.bmm(query.transpose(1, 2), key)
        attention = F.softmax(energy, dim=-1)
        out = torch.bmm(x_patch, attention)
        out = out.view(batch_size, c, h // self.patch_size, w // self.patch_size)
        out = F.interpolate(out, size=(h, w), mode='nearest')
        return self.gamma * out


class _AttentionModule(nn.Module):
    def __init__(self, n_feats):
        super(_AttentionModule, self).__init__()

        self.spatial_atten = Spatial_Atten(n_feats)
        self.channel_atten = Channel_Atten(n_feats)

        self.fusion = nn.Sequential(
            nn.Conv2d(n_feats , n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        spatial_att = self.spatial_atten(x)
        channel_att = self.channel_atten(x)

        h = self.fusion(spatial_att+ channel_att)+x
        return h



# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////   BFFN
# #

class BFFN(nn.Module):
    def __init__(self, n_feats=64):  # or pass n_feats from opt
        super(BFFN, self).__init__()

        self.layer0 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False), nn.LeakyReLU())
        self.layer1 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False), nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False), nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False), nn.LeakyReLU())

        self.down4 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False), nn.LeakyReLU())
        self.down3 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False), nn.LeakyReLU())
        self.down2 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False), nn.LeakyReLU())
        self.down1 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False), nn.LeakyReLU())

        self.refine3_hl = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False), nn.LeakyReLU())
        self.refine2_hl = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False), nn.LeakyReLU())
        self.refine1_hl = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False), nn.LeakyReLU())

        self.attention3_hl = _AttentionModule(n_feats)
        self.attention2_hl = _AttentionModule(n_feats)
        self.attention1_hl = _AttentionModule(n_feats)

        self.refine2_lh = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False), nn.LeakyReLU())
        self.refine4_lh = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False), nn.LeakyReLU())
        self.refine3_lh = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=False), nn.LeakyReLU())

        self.attention2_lh = _AttentionModule(n_feats)
        self.attention3_lh = _AttentionModule(n_feats)
        self.attention4_lh = _AttentionModule(n_feats)

        self.predict = nn.Sequential(nn.Conv2d(n_feats, n_feats, 1, padding=0, bias=False), nn.LeakyReLU())
        self.end = nn.Sequential(nn.Conv2d(n_feats, n_feats, 1, padding=0, bias=False), nn.LeakyReLU())

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down4 = self.down4(layer4)
        down3 = self.down3(layer3)
        down2 = self.down2(layer2)
        down1 = self.down1(layer1)

        down4 = F.interpolate(down4, size=down3.shape[2:], mode='bilinear', align_corners=False)
        refine3_hl = F.relu(self.refine3_hl(down4 + down3) + down4, inplace=True)
        refine3_hl = F.relu(self.attention3_hl(refine3_hl) + down3, inplace=True)
        refine3_hl = F.interpolate(refine3_hl, size=down2.shape[2:], mode='bilinear', align_corners=False)

        refine2_hl = F.relu(self.refine2_hl(refine3_hl + down2) + refine3_hl, inplace=True)
        refine2_hl = F.relu(self.attention2_hl(refine2_hl) + down2, inplace=True)
        refine2_hl = F.interpolate(refine2_hl, size=down1.shape[2:], mode='bilinear', align_corners=False)

        refine1_hl = F.relu(self.refine1_hl(refine2_hl + down1) + refine2_hl, inplace=True)
        refine1_hl = F.relu(self.attention1_hl(refine1_hl) + refine1_hl, inplace=True)

        down2 = F.interpolate(down2, size=down1.shape[2:], mode='bilinear', align_corners=False)
        refine2_lh = F.relu(self.refine2_lh(down1 + down2) + down1, inplace=True)
        refine2_lh = F.relu(self.attention2_lh(refine2_lh) + refine2_lh, inplace=True)

        down3 = F.interpolate(down3, size=down1.shape[2:], mode='bilinear', align_corners=False)
        refine3_lh = F.relu(self.refine3_lh(refine2_lh + down3) + refine2_lh, inplace=True)
        refine3_lh = F.relu(self.attention3_lh(refine3_lh) + refine3_lh, inplace=True)

        down4 = F.interpolate(down4, size=down1.shape[2:], mode='bilinear', align_corners=False)
        refine4_lh = F.relu(self.refine4_lh(refine3_lh + down4) + refine3_lh, inplace=True)
        refine4_lh = F.relu(self.attention4_lh(refine4_lh) + refine4_lh, inplace=True)

        fuse = self.attention2_lh(refine1_hl + refine4_lh)
        fuse = F.interpolate(fuse, size=x.shape[2:], mode='bilinear', align_corners=False)

        predict_end = self.end(fuse) + x
        return predict_end
