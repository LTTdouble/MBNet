import math

from torch.nn import init as init
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    wn = lambda x: torch.nn.utils.weight_norm(x)
    return wn(nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias))
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class ResAttentionBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttentionBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        m.append(CALayer(n_feats, 16))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res



class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False ,act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))

            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def _to_4d_tensor(x, depth_stride=None):
    """Converts a 5d tensor to 4d by stacking
    the batch and depth dimensions."""
    x = x.transpose(0, 2)  # swap batch and depth dimensions: NxCxDxHxW => DxCxNxHxW
    if depth_stride:
        x = x[::depth_stride]  # downsample feature maps along depth dimension
    depth = x.size()[0]
    x = x.permute(2, 0, 1, 3, 4)  # DxCxNxHxW => NxDxCxHxW
    x = torch.split(x, 1, dim=0)  # split along batch dimension: NxDxCxHxW => N*[1xDxCxHxW]
    x = torch.cat(x, 1)  # concatenate along depth dimension: N*[1xDxCxHxW] => 1x(N*D)xCxHxW
    x = x.squeeze(0)  # 1x(N*D)xCxHxW => (N*D)xCxHxW
    return x, depth


def _to_5d_tensor(x, depth):
    """Converts a 4d tensor back to 5d by splitting
    the batch dimension to restore the depth dimension."""
    x = torch.split(x, depth)  # (N*D)xCxHxW => N*[DxCxHxW]
    x = torch.stack(x, dim=0)  # re-instate the batch dimension: NxDxCxHxW
    x = x.transpose(1, 2)  # swap back depth and channel dimensions: NxDxCxHxW => NxCxDxHxW
    return x


# class TreeDBlock(nn.Module):
#     def __init__(self, n_feats):
#         super(TreeDBlock, self).__init__()
#
#         self.conv3D_1 = nn.Conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
#         self.conv3D_2= nn.Conv3d(n_feats, n_feats, kernel_size=(3,1, 1), stride=1, padding=(1,0, 0))
#         # self.conv3D_3 = nn.Conv3d(n_feats, n_feats, kernel_size=(3,3, 1), stride=1, padding=(1,1, 0))
#
#         self.ReLU = nn.ReLU(inplace=True)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#     def forward(self, x):
#         x1 = self.conv3D_1(x)
#         x1= self.ReLU(x1)
#         x1 = self.conv3D_2(x1)
#         # x1 = self.lrelu(x1)
#         # x1 = self.conv3D_3(x1)
#         # x1=x+x1
#         return x1

class BasicConv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=(0, 0, 0), use_relu=True):
        super(BasicConv3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channel, out_channel,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.use_relu = use_relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class TreeDBlock(nn.Module):
    def __init__(self, cin, cout, use_relu, fea_num):
        super(TreeDBlock, self).__init__()

        self.spatiallayer = nn.Conv3d(cout, cout, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                        bias=False)
        self.spectralayer = nn.Conv3d(cout, cout, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0),
                                        bias=False)
        self.Conv_mixdence = BasicConv3d(cout, cout, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                        use_relu=use_relu)

        # self.spatiallayer = BasicConv3d(cout, cout, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
        #                                 use_relu=use_relu)
        # self.spectralayer = BasicConv3d(cout, cout, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0),
        #                                 use_relu=use_relu)

        # self.Conv_mixdence = nn.Conv3d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False)
        #
        # self.Conv_mixdence = nn.Sequential(
        #     nn.Conv3d(cin, cout, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
        #     nn.Conv3d(cin, cout, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False))

        self.relu = nn.ReLU(inplace=True)

        self.fea_num = fea_num

        self.component_num = cout
        self.feat_weight = nn.Parameter(torch.rand(fea_num * 64), requires_grad=True)  # [N,*,c,u,v,h,w]
        self.component_weight1 = nn.Parameter(torch.rand(self.component_num), requires_grad=True)  # [N,*,c,,,]
        self.component_weight2 = nn.Parameter(torch.rand(self.component_num), requires_grad=True)  # [N,*,c,,,]
        self.temperature_1 = 0.2
        self.temperature_2 = 0.2

    def forward(self, x, epoch):

        t1 = 1.0
        t2 = 1.0
        if epoch <= 30:  # T  1 ==> 0.1
            self.temperature_1 = t1 * (1 - epoch / 35)
            self.temperature_2 = t2 * (1 - epoch / 35)
        else:
            self.temperature_1 = 0.05
            self.temperature_2 = 0.05

        if (self.fea_num > 1):
            x = torch.cat(x, dim=1)  # [fea_num B C H W]

            [B, L, C, H, W] = x.shape

            feat_weight = self.feat_weight.clamp(0.02, 0.98)
            feat_weight = feat_weight[None, :, None, None, None]
            # p shape[fea_num 1 1 1 1]
            # noise r1 r2
            noise_feat_r1 = torch.rand((B, self.fea_num * 64))[:, :, None, None, None].cuda()  ##[dence_num,N,1,1,1,1]
            noise_feat_r2 = torch.rand((B, self.fea_num * 64))[:, :, None, None, None].cuda()
            noise_feat_logits = torch.log(torch.log(noise_feat_r1) / torch.log(noise_feat_r2))
            feat_weight_soft = torch.sigmoid(
                (torch.log(feat_weight / (1 - feat_weight)) + noise_feat_logits) / self.temperature_1)
            feat_logits = feat_weight_soft

            x = x * feat_logits
        # else:
        #     x = torch.cat(x, 1)

        # # SELECT NETWOKR
        component_weight1 = self.component_weight1.clamp(0.02, 0.98)
        component_weight1 = component_weight1[None, :, None, None, None]
        component_weight2 = self.component_weight2.clamp(0.02, 0.98)
        component_weight2 = component_weight2[None, :, None, None, None]

        [B, L, C, H, W] = x.shape

        # s2
        noise_component_r1 = torch.rand((B, self.component_num))[:, :, None, None, None].cuda()  ##[dence_num,N,1,1,1,1]
        noise_component_r2 = torch.rand((B, self.component_num))[:, :, None, None, None].cuda()
        noise_component_logits1 = torch.log(torch.log(noise_component_r1) / torch.log(noise_component_r2))
        component_weight_gumbel1 = torch.sigmoid(
            (torch.log(component_weight1 / (1 - component_weight1)) + noise_component_logits1) / self.temperature_2)
        logits2 = component_weight_gumbel1

        # s3
        noise_component_r3 = torch.rand((B, self.component_num))[:, :, None, None, None].cuda()  ##[dence_num,N,1,1,1,1]
        noise_component_r4 = torch.rand((B, self.component_num))[:, :, None, None, None].cuda()
        noise_component_logits2 = torch.log(torch.log(noise_component_r3) / torch.log(noise_component_r4))
        component_weight_gumbel2 = torch.sigmoid(
            (torch.log(component_weight2 / (1 - component_weight2)) + noise_component_logits2) / self.temperature_2)
        logits3 = component_weight_gumbel2

        output = self.relu(self.Conv_mixdence(x))

        output = self.spectralayer(output) * logits2 + output
        output = self.spatiallayer(output) * logits3 + output+x

        return output



def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


import torch.nn.functional as F

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()

        # self.spatial = BasicConv(2, 1, kernel_size=3, stride=1, padding=(kernel_size - 1) // 2, relu=False)

        self.spatial = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=(kernel_size - 1) // 2),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


##########################################################################
## ------ Channel Attention --------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class _AttentionModule(nn.Module):
    def __init__(self,n_feats):
        super(_AttentionModule, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(2*n_feats, n_feats, 1, padding=0,bias=False),  nn.ReLU(),

        )
        self.block2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1,bias=False),  nn.ReLU(),

        )
        self.SA = spatial_attn_layer(kernel_size=3)
        self.CA = ca_layer(n_feats, reduction=8, bias=True)
        self.conv1x1 = nn.Conv2d(2*n_feats , n_feats, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        block1 = F.relu(self.block1(x) , True)
        block2 = F.relu(self.block2(block1) + block1, True)
        sa_branch = self.SA(block2)
        ca_branch = self.CA(block2)
        h = self.conv1x1(torch.cat([sa_branch, ca_branch], dim=1))

        # h = sa_branch+ ca_branch
        # h = self.block1(h)+block2
        return h