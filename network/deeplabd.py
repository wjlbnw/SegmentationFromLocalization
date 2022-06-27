import torch
from torch import nn
import numpy as np
import math
# from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
#
# BatchNorm2d = SynchronizedBatchNorm2d



import torch.nn.functional as F
class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        if first_dilation == None: first_dilation = dilation

        self.bn_branch2a = nn.BatchNorm2d(in_channels)

        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
                                       padding=first_dilation, dilation=first_dilation, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(mid_channels)

        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class ResBlock_bot(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.):
        super(ResBlock_bot, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        self.bn_branch2a = nn.BatchNorm2d(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels//4, 1, stride, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(out_channels//4)
        self.dropout_2b1 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b1 = nn.Conv2d(out_channels//4, out_channels//2, 3, padding=dilation, dilation=dilation, bias=False)

        self.bn_branch2b2 = nn.BatchNorm2d(out_channels//2)
        self.dropout_2b2 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels//2, out_channels, 1, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2)

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = self.bn_branch2b2(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

#
# class ASPP_module(nn.Module):  # ASpp模块的组成
#     def __init__(self, inplanes, planes, dilation):
#         super(ASPP_module, self).__init__()
#         if dilation == 1:
#             kernel_size = 1
#             padding = 0
#         else:
#             kernel_size = 3
#             padding = dilation
#         self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
#                                             stride=1, padding=padding, dilation=dilation, bias=False)
#         # self.bn = BatchNorm2d(planes)
#         self.bn = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU()
#         self._init_weight()
#
#     def forward(self, x):
#         x = self.atrous_convolution(x)
#         x = self.bn(x)
#         return self.relu(x)
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#
# # 正式开始deeplabv3+的结构组成
# class DeepLabv3_plus(nn.Module):
#     def __init__(self, nInputChannels=3, n_classes=21, os=16, pretrained=False, freeze_bn=False, _print=True):
#         if _print:
#             print("Constructing DeepLabv3+ model...")
#             print("Backbone: Resnet-101")
#             print("Number of classes: {}".format(n_classes))
#             print("Output stride: {}".format(os))
#             print("Number of Input Channels: {}".format(nInputChannels))
#         super(DeepLabv3_plus, self).__init__()
#
#         # Atrous Conv  首先获得从resnet101中提取的features map
#         self.resnet_features = ResNet101(nInputChannels, os, pretrained=pretrained)
#
#         # ASPP,挑选参数
#         if os == 16:
#             dilations = [1, 6, 12, 18]
#         elif os == 8:
#             dilations = [1, 12, 24, 36]
#         else:
#             raise NotImplementedError
#         # 四个不同带洞卷积的设置，获取不同感受野
#         self.aspp1 = ASPP_module(2048, 256, dilation=dilations[0])
#         self.aspp2 = ASPP_module(2048, 256, dilation=dilations[1])
#         self.aspp3 = ASPP_module(2048, 256, dilation=dilations[2])
#         self.aspp4 = ASPP_module(2048, 256, dilation=dilations[3])
#         self.relu = nn.ReLU()
#
#         # 全局平均池化层的设置
#         self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
#                                              nn.Conv2d(2048, 256, 1, stride=1, bias=False),
#                                              BatchNorm2d(256),
#                                              nn.ReLU())
#
#         self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
#         self.bn1 = BatchNorm2d(256)
#
#         # adopt [1x1, 48] for channel reduction.
#         self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
#         self.bn2 = BatchNorm2d(48)
#         # 结构图中的解码部分的最后一个3*3的卷积块
#         self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        BatchNorm2d(256),
#                                        nn.ReLU(),
#                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#                                        BatchNorm2d(256),
#                                        nn.ReLU(),
#                                        nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
#
#         if freeze_bn:
#             self._freeze_bn()
#
#     # 前向传播
#
#     def forward(self, input):
#         x, low_level_features = self.resnet_features(input)
#         x1 = self.aspp1(x)
#         x2 = self.aspp2(x)
#         x3 = self.aspp3(x)
#         x4 = self.aspp4(x)
#         x5 = self.global_avg_pool(x)
#         x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
#         # 把四个ASPP模块以及全局池化层拼接起来
#         x = torch.cat((x1, x2, x3, x4, x5), dim=1)
#         # 上采样
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = F.upsample(x, size=(int(math.ceil(input.size()[-2] / 4)),
#                                 int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)
#
#         low_level_features = self.conv2(low_level_features)
#         low_level_features = self.bn2(low_level_features)
#         low_level_features = self.relu(low_level_features)
#
#         # 拼接低层次的特征，然后再通过插值获取原图大小的结果
#         x = torch.cat((x, low_level_features), dim=1)
#         x = self.last_conv(x)
#         # 实现插值和上采样
#         x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
#         return x
#
#     def _freeze_bn(self):
#         for m in self.modules():
#             if isinstance(m, BatchNorm2d):
#                 m.eval()
#
#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#
# def get_1x_lr_params(model):
#     """
#     This generator returns all the parameters of the net except for
#     the last classification layer. Note that for each batchnorm layer,
#     requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
#     any batchnorm parameter
#     """
#     b = [model.resnet_features]
#     for i in range(len(b)):
#         for k in b[i].parameters():
#             if k.requires_grad:
#                 yield k
#
#
# def get_10x_lr_params(model):
#     """
#     This generator returns all the parameters for the last layer of the net,
#     which does the classification of pixel into classes
#     """
#     b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
#     for j in range(len(b)):
#         for k in b[j].parameters():
#             if k.requires_grad:
#                 yield k
#
#


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5)

        self.bn7 = nn.BatchNorm2d(4096)

        self.not_training = [self.conv1a]

        self.normalize = Normalize()

        return

    def forward(self, x):
        return self.forward_as_dict(x)['conv6']

    def forward_as_dict(self, x):

        x = self.conv1a(x)
        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)
        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)
        # print('after b3:', x.shape)
        #x = self.b4(x)
        x, conv3 = self.b4(x, get_x_bn_relu=True)
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x, get_x_bn_relu=True)
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)

        x = self.b7(x)
        conv6 = F.relu(self.bn7(x))

        return dict({'conv3': conv3, 'conv4': conv4, 'conv5': conv5, 'conv6': conv6})


    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

def convert_mxnet_to_torch(filename):
    import mxnet

    save_dict = mxnet.nd.load(filename)

    renamed_dict = dict()

    bn_param_mx_pt = {'beta': 'bias', 'gamma': 'weight', 'mean': 'running_mean', 'var': 'running_var'}

    for k, v in save_dict.items():

        v = torch.from_numpy(v.asnumpy())
        toks = k.split('_')

        if 'conv1a' in toks[0]:
            renamed_dict['conv1a.weight'] = v

        elif 'linear1000' in toks[0]:
            pass

        elif 'branch' in toks[1]:

            pt_name = []

            if toks[0][-1] != 'a':
                pt_name.append('b' + toks[0][-3] + '_' + toks[0][-1])
            else:
                pt_name.append('b' + toks[0][-2])

            if 'res' in toks[0]:
                layer_type = 'conv'
                last_name = 'weight'

            else:  # 'bn' in toks[0]:
                layer_type = 'bn'
                last_name = bn_param_mx_pt[toks[-1]]

            pt_name.append(layer_type + '_' + toks[1])

            pt_name.append(last_name)

            torch_name = '.'.join(pt_name)
            renamed_dict[torch_name] = v

        else:
            last_name = bn_param_mx_pt[toks[-1]]
            renamed_dict['bn7.' + last_name] = v

    return renamed_dict

