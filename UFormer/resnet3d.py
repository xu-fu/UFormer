import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from timm.models.layers import DropPath, trunc_normal_


__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class NormalLayer(nn.Module):
    def __init__(self, dim, groups=16):
        super().__init__()
        # 归一化层
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=dim)

    def forward(self, x):
        x = self.group_norm(x)
        return x
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, kernel_size=3, stride=2):
        super(PatchEmbed, self).__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size, stride, padding=kernel_size // stride, bias=False)
        self.norm = nn.GroupNorm(1, embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x
        
    
class Myblock(nn.Module):
    def __init__(self, modulelist):
        super(Myblock, self).__init__()
        self.layers = modulelist

    def forward(self, x, d, h, w):
        for layer in self.layers:
            x = layer(x, d, h, w)
        return x
   
class Myblockcross(nn.Module):
    def __init__(self, modulelist):
        super(Myblockcross, self).__init__()
        self.layers = modulelist

    def forward(self, x, x2, d, h, w):
        for layer in self.layers:
            x = layer(x, x2, d, h, w)
        return x


class Res3TransCross(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes,
                 ipt_dim=1,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(Res3TransCross, self).__init__()
        self.conv1 = nn.Conv3d(ipt_dim,64,kernel_size=7,stride=(2, 2, 2),padding=(3, 3, 3),bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)

        dpr = [x.item() for x in torch.linspace(0, 0, sum(layers))]  # stochastic depth decay rule
        self.layer3 = Myblock(nn.ModuleList([AttentionBlock(
                                    dim=512, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0., drop_path=dpr[layers[0]+layers[1]+j], norm_layer=nn.LayerNorm,
                                    sr_ratio=2, linear=False)
                                    for j in range(layers[2])]))
        self.layer4 = Myblock(nn.ModuleList([AttentionBlock(
                                    dim=1024, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0., drop_path=dpr[layers[0]+layers[1]+layers[2]+j], norm_layer=nn.LayerNorm,
                                    sr_ratio=1, linear=False)
                                    for j in range(layers[3])]))
       
        self.down3 = nn.Conv3d(512, 512*2, 3, 2, padding=3//2, bias=False)
        # self.down3 = PatchEmbed(512, 512*2, 3, 2)

        self.conv1_2 = nn.Conv3d(ipt_dim,64,kernel_size=7,stride=(2, 2, 2),padding=(3, 3, 3),bias=False)
        self.bn1_2 = nn.BatchNorm3d(64)
        self.relu_2 = nn.ReLU(inplace=True)
        self.maxpool_2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.inplanes = 64
        self.layer1_2 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2_2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)

        self.layer3_2 = Myblock(nn.ModuleList([AttentionBlock(
                                    dim=512, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0., drop_path=dpr[layers[0]+layers[1]+j], norm_layer=nn.LayerNorm,
                                    sr_ratio=2, linear=False)
                                    for j in range(layers[2])]))
        self.layer4_2 = Myblock(nn.ModuleList([AttentionBlock(
                                    dim=1024, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0., drop_path=dpr[layers[0]+layers[1]+layers[2]+j], norm_layer=nn.LayerNorm,
                                    sr_ratio=1, linear=False)
                                    for j in range(layers[3])]))
        
        self.down3_2 = nn.Conv3d(512, 512*2, 3, 2, padding=3//2, bias=False)


        self.conv1_3 = nn.Conv3d(ipt_dim,64,kernel_size=7,stride=(2, 2, 2),padding=(3, 3, 3),bias=False)
        self.bn1_3 = nn.BatchNorm3d(64)
        self.relu_3 = nn.ReLU(inplace=True)
        self.maxpool_3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.inplanes = 64
        self.layer1_3 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2_3 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)

        self.layer3_3 = Myblock(nn.ModuleList([AttentionBlock(
                                    dim=512, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0., drop_path=dpr[layers[0]+layers[1]+j], norm_layer=nn.LayerNorm,
                                    sr_ratio=2, linear=False)
                                    for j in range(layers[2])]))
        self.layer4_3 = Myblock(nn.ModuleList([AttentionBlock(
                                    dim=1024, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0., drop_path=dpr[layers[0]+layers[1]+layers[2]+j], norm_layer=nn.LayerNorm,
                                    sr_ratio=1, linear=False)
                                    for j in range(layers[3])]))
        
        self.down3_3 = nn.Conv3d(512, 512*2, 3, 2, padding=3//2, bias=False)
        # self.down3_3 = PatchEmbed(512, 512*2, 3, 2)

        self.crosslayer1 = Myblockcross(nn.ModuleList([CrossAttentionBlock(
                                    dim=1024, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                                    sr_ratio=1, linear=False)
                                    for j in range(1)]))
        self.crosslayer2 = Myblockcross(nn.ModuleList([CrossAttentionBlock(
                                    dim=1024, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                                    sr_ratio=1, linear=False)
                                    for j in range(1)]))
        self.crosslayer3 = Myblockcross(nn.ModuleList([CrossAttentionBlock(
                                    dim=1024, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                                    sr_ratio=1, linear=False)
                                    for j in range(1)]))
        

        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(1 * 256 * block.expansion, num_classes)
        

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, xt2, xdwi, xadc):
        xt2 = self.conv1(xt2) # 1, 16, 128, 128
        xt2 = self.bn1(xt2)
        xt2 = self.relu(xt2)
        xt2 = self.maxpool(xt2) # 64, 4, 32, 32
        # xt2 = self.d1(xt2)
        xt2 = self.layer1(xt2) # 256, 4, 32, 32
        xt2 = self.layer2(xt2) #512, 2, 16, 16
        # xt2 = self.d1(xt2)
        # xt2 = self.layer3(xt2) #1024, 1, 8, 8
        xt2 = xt2.view(xt2.shape[0], xt2.shape[1], -1).permute(0, 2, 1) #b n c 
        xt2 = self.layer3(xt2, 2, 16, 16).permute(0, 2, 1) #b c n 
        xt2 = xt2.view(xt2.shape[0], xt2.shape[1], 2, 16, 16)
        xt2 = self.down3(xt2)
        xt2 = xt2.view(xt2.shape[0], xt2.shape[1], -1).permute(0, 2, 1) #b n c 
        xt2 = self.layer4(xt2, 1, 8, 8).permute(0, 2, 1) #b c n 
        xt2 = xt2.view(xt2.shape[0], xt2.shape[1], 1, 8, 8)
        # xt2 = self.d1(xt2)
        # xt2 = self.layer4(xt2) #2048, 1, 8, 8

        xdwi = self.conv1_2(xdwi)
        xdwi = self.bn1_2(xdwi)
        xdwi = self.relu_2(xdwi)
        xdwi = self.maxpool_2(xdwi)

        xdwi = self.layer1_2(xdwi)
        xdwi = self.layer2_2(xdwi)

        xdwi = xdwi.view(xdwi.shape[0], xdwi.shape[1], -1).permute(0, 2, 1)
        xdwi = self.layer3_2(xdwi, 2, 16, 16).permute(0, 2, 1)
        xdwi = xdwi.view(xdwi.shape[0], xdwi.shape[1], 2, 16, 16)
        xdwi = self.down3_2(xdwi)
        xdwi = xdwi.view(xdwi.shape[0], xdwi.shape[1], -1).permute(0, 2, 1)
        xdwi = self.layer4_2(xdwi, 1, 8, 8).permute(0, 2, 1)
        xdwi = xdwi.view(xdwi.shape[0], xdwi.shape[1], 1, 8, 8)


        xadc = self.conv1_3(xadc)
        xadc = self.bn1_3(xadc)
        xadc = self.relu_3(xadc)
        xadc = self.maxpool_3(xadc)

        xadc = self.layer1_3(xadc)
        xadc = self.layer2_3(xadc)

        xadc = xadc.view(xadc.shape[0], xadc.shape[1], -1).permute(0, 2, 1)
        xadc = self.layer3_3(xadc, 2, 16, 16).permute(0, 2, 1)
        xadc = xadc.view(xadc.shape[0], xadc.shape[1], 2, 16, 16)
        xadc = self.down3_3(xadc)
        xadc = xadc.view(xadc.shape[0], xadc.shape[1], -1).permute(0, 2, 1)
        xadc = self.layer4_3(xadc, 1, 8, 8).permute(0, 2, 1)
        xadc = xadc.view(xadc.shape[0], xadc.shape[1], 1, 8, 8)



        xadd = xt2 + xdwi + xadc
        xadd = xadd.view(xadd.shape[0], xadd.shape[1], -1).permute(0, 2, 1)
        # xadd = self.layer5(xadd, 1, 8, 8)
        xt2 = xt2.view(xt2.shape[0], xt2.shape[1], -1).permute(0, 2, 1)
        xdwi = xdwi.view(xdwi.shape[0], xdwi.shape[1], -1).permute(0, 2, 1)
        xadc = xadc.view(xadc.shape[0], xadc.shape[1], -1).permute(0, 2, 1)

        xaddt2 = self.crosslayer1(xadd, xt2, 1, 8, 8).permute(0, 2, 1)
        xadddwi = self.crosslayer2(xadd, xdwi, 1, 8, 8).permute(0, 2, 1)
        xaddadc = self.crosslayer3(xadd, xadc, 1, 8, 8).permute(0, 2, 1)

        xaddt2 = xaddt2.view(xaddt2.shape[0], xaddt2.shape[1], 1, 8, 8)
        xadddwi = xadddwi.view(xadddwi.shape[0], xadddwi.shape[1], 1, 8, 8)
        xaddadc = xaddadc.view(xaddadc.shape[0], xaddadc.shape[1], 1, 8, 8)

        x = xaddt2 + xadddwi + xaddadc


        xout = x
        x = self.avgpool(x)
        features = torch.flatten(x,1)
        # x = self.drop(x)
        x = self.classifier(features)
        # x = self.prelu(x)
        # features2 = x
        # x = self.classifier2(x)

        return (xt2, xdwi, xadc), features, x




class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                #  sample_input_D,
                #  sample_input_H,
                #  sample_input_W,
                 num_classes,
                 ipt_dim=1,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(ipt_dim,64,kernel_size=7,stride=(2, 2, 2),padding=(3, 3, 3),bias=False)
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2) #2
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4) #4

        

        # self.avgpool = nn.AdaptiveAvgPool3d(1)
        # self.classifier = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.conv_seg(x)
        xout = x

        # x = self.avgpool(x)
        # features = torch.flatten(x,1)

        # x = self.classifier(features)

        return xout, x


class Res3Net(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes,
                 ipt_dim=1,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(Res3Net, self).__init__()
        self.conv1 = nn.Conv3d(ipt_dim,64,kernel_size=7,stride=(2, 2, 2),padding=(3, 3, 3),bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2, dilation=1)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=1)

        self.conv1_2 = nn.Conv3d(ipt_dim,64,kernel_size=7,stride=(2, 2, 2),padding=(3, 3, 3),bias=False)
        self.bn1_2 = nn.BatchNorm3d(64)
        self.relu_2 = nn.ReLU(inplace=True)
        self.maxpool_2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.inplanes = 64
        self.layer1_2 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2_2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3_2 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2, dilation=1)
        self.layer4_2 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=1)

        self.conv1_3 = nn.Conv3d(ipt_dim,64,kernel_size=7,stride=(2, 2, 2),padding=(3, 3, 3),bias=False)
        self.bn1_3 = nn.BatchNorm3d(64)
        self.relu_3 = nn.ReLU(inplace=True)
        self.maxpool_3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.inplanes = 64
        self.layer1_3 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2_3 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3_3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2, dilation=1)
        self.layer4_3 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=1)

        self.crosslayer1 = Myblockcross(nn.ModuleList([CrossAttentionBlock(
                                    dim=1024*2, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                                    sr_ratio=1, linear=False)
                                    for j in range(1)]))
        self.crosslayer2 = Myblockcross(nn.ModuleList([CrossAttentionBlock(
                                    dim=1024*2, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                                    sr_ratio=1, linear=False)
                                    for j in range(1)]))
        self.crosslayer3 = Myblockcross(nn.ModuleList([CrossAttentionBlock(
                                    dim=1024*2, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                    drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                                    sr_ratio=1, linear=False)
                                    for j in range(1)]))

        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)
        

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, xt2, xdwi, xadc):
        xt2 = self.conv1(xt2) # 1, 16, 128, 128
        xt2 = self.bn1(xt2)
        xt2 = self.relu(xt2)
        xt2 = self.maxpool(xt2) # 64, 4, 32, 32
        # xt2 = self.d1(xt2)
        xt2 = self.layer1(xt2) # 256, 4, 32, 32
        xt2 = self.layer2(xt2) #512, 2, 16, 16
        # xt2 = self.d1(xt2)
        xt2 = self.layer3(xt2) #1024, 1, 8, 8
        # xt2 = self.d1(xt2)
        xt2 = self.layer4(xt2) #2048, 1, 8, 8

        xdwi = self.conv1_2(xdwi)
        xdwi = self.bn1_2(xdwi)
        xdwi = self.relu_2(xdwi)
        xdwi = self.maxpool_2(xdwi)
        # xdwi = self.d2(xdwi)
        xdwi = self.layer1_2(xdwi)
        xdwi = self.layer2_2(xdwi)
        # xdwi = self.d2(xdwi)
        xdwi = self.layer3_2(xdwi)
        # xdwi = self.d2(xdwi)
        xdwi = self.layer4_2(xdwi)

        xadc = self.conv1_3(xadc)
        xadc = self.bn1_3(xadc)
        xadc = self.relu_3(xadc)
        xadc = self.maxpool_3(xadc)
        # xadc = self.d3(xadc)
        xadc = self.layer1_3(xadc)
        xadc = self.layer2_3(xadc)
        # xadc = self.d3(xadc)
        xadc = self.layer3_3(xadc)
        # xadc = self.d3(xadc)
        xadc = self.layer4_3(xadc)

        # x = (xt2 + xdwi + xadc)/3
        # # x = torch.cat([xt2 , xdwi , xadc], dim=1)
        # # x = self.reluf(self.bnf(self.convf(x)))
        # 相加后交叉注意力
        xadd = xt2 + xdwi + xadc
        xadd = xadd.view(xadd.shape[0], xadd.shape[1], -1).permute(0, 2, 1)
        xt2 = xt2.view(xt2.shape[0], xt2.shape[1], -1).permute(0, 2, 1)
        xdwi = xdwi.view(xdwi.shape[0], xdwi.shape[1], -1).permute(0, 2, 1)
        xadc = xadc.view(xadc.shape[0], xadc.shape[1], -1).permute(0, 2, 1)

        xaddt2 = self.crosslayer1(xadd, xt2, 1, 8, 8).permute(0, 2, 1)
        xadddwi = self.crosslayer2(xadd, xdwi, 1, 8, 8).permute(0, 2, 1)
        xaddadc = self.crosslayer3(xadd, xadc, 1, 8, 8).permute(0, 2, 1)

        xaddt2 = xaddt2.view(xaddt2.shape[0], xaddt2.shape[1], 1, 8, 8)
        xadddwi = xadddwi.view(xadddwi.shape[0], xadddwi.shape[1], 1, 8, 8)
        xaddadc = xaddadc.view(xaddadc.shape[0], xaddadc.shape[1], 1, 8, 8)

        x = xaddt2 + xadddwi + xaddadc

        xout = x
        x = self.avgpool(x)
        features = torch.flatten(x,1)
        # x = self.drop(x)
        x = self.classifier(features)
        # x = self.prelu(x)
        # features2 = x
        # x = self.classifier2(x)

        return (xt2, xdwi, xadc), features, x


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    # model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    model = Res3Net(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    # model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    model = Res3Net(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(pretrain=False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model = Res3Net(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrain:
        weight_dir = '/resnet_34.pth'
        print("load pre weight...",weight_dir.split('/')[-1])
        checkpoint = torch.load(weight_dir)
        state_dict = checkpoint['state_dict']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        model.load_state_dict(unParalled_state_dict, strict=False) 
    
    return model



def restranscross503(pretrain=False, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = Res3TransCross(Bottleneck, [3, 4, 12, 3], **kwargs)
    if pretrain:
        weight_dir = '/resnet.pth'
        print("load pre weight...", weight_dir.split('/')[-1])
        checkpoint = torch.load(weight_dir)
        state_dict = checkpoint['state_dict']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            unParalled_state_dict[key.replace("module.", "").replace('.weight', '_2.weight')] = state_dict[key]
            unParalled_state_dict[key.replace("module.", "").replace('.weight', '_3.weight')] = state_dict[key]
        model.load_state_dict(unParalled_state_dict, strict=False) 
    
    return model



def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = Res3Net(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

