import os
import math
import time
from math import floor, ceil
#from pytorch_pretrained_vit import PreTrainModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms

name = 'resnet101'
path = os.path.join(os.getcwd(), 'pretrained_models', name+'.pth')
name1 = 'B_16'
path1 = os.path.join(os.getcwd(), 'pretrained_models', name1+'.pth')

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, dilation, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample   #对输入特征图大小进行减半处理
        self.stride = stride

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

    def __init__(self, inplanes, planes, dilation, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, dilation=dilation, kernel_size=3, stride=stride,
                            padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 网络输入部分
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 中间卷积部分
        self.layer1 = self._make_layer(block, 64, layers[0], dilation=1)
        self.layer2 = self._make_layer(block, 128, layers[1], dilation=1, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], dilation=1, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], dilation=2, stride=1)
        # 平均池化和全连接层
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, dilation, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, dilation=1, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes, dilation=dilation))

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

        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return x


class BaseResnet(nn.Module):
    def __init__(self, res_path):
        super(BaseResnet, self).__init__()
        self.model = models.resnet101(pretrained=False)
        self.model.load_state_dict(torch.load(res_path))
        print("Loaded pretrained ResNet101 weights.")
        #self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth'))
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        # x = self.model.avgpool(x)
        # x = x.view(x.size(0), x.size(1))
        return x


class SpatialPyramidPooling2d(nn.Module):
    def __init__(self, levels=[1, 2, 4, 6], pool_type='max_pool'):
        super(SpatialPyramidPooling2d, self).__init__()
        assert pool_type in ['max_pool', 'avg_pool'], "Wrong Pooling Type!"
        self.levels = levels
        self.pools = []
        if pool_type == 'max_pool':
            for i, level in enumerate(levels):
                self.pools.append(nn.AdaptiveAvgPool2d((level, level)))
        elif pool_type == 'avg_pool':
            for i ,level in enumerate(levels):
                self.pools.append(nn.AdaptiveMaxPool2d((level, level)))
        self.pool_type = pool_type
        
    def forward(self, x):
        N, C, H, W = x.size()
        for i, pool in enumerate(self.pools):
            tensor = self.pools[i](x)
            tensor = F.interpolate(tensor, size=(H, W), mode='bilinear', align_corners=False)

            if i == 0:
                res = tensor
            else:
                res = torch.cat((res, tensor), 1)
        return res


class Decoder2D(nn.Module):
    def __init__(self, size, in_channels, out_channels=11, planes=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, planes, 3, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.final_out = nn.Conv2d(planes, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.decoder(x)
        x = F.interpolate(x, size=x.size()[2:], mode='bilinear', align_corners=False)
        x = self.final_out(x)
        return x

'''
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels=3, planes=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, out_channels=planes, kernel_size=3, stride=17, padding=1, output_padding=0, bias=False
            ),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.final_out = nn.Conv2d(in_channels=planes, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.decoder(x)
        x = self.final_out(x)
        return x
'''

class BasicParseNet(nn.Module):
    def __init__(self, cfg):
        super(BasicParseNet, self).__init__()
        self.resnet = ResNet(Bottleneck, [3, 4, 23, 3])
        self.planes = cfg.MODEL.DIM_MODEL
        self.out_channels = cfg.MODEL.NUM_SEGMENTS
        if cfg.MODEL.PRETRAINED:
            self.resnet.load_state_dict(torch.load(cfg.MODEL.PRETRAINED_PATH))
            print("Loaded pretrained ResNet101 weights.")
        self.sppnet = SpatialPyramidPooling2d(levels=cfg.MODEL.NUM_POOL_LEVELS, pool_type=cfg.MODEL.POOL_TYPE)
        #self.conv_pre = nn.Conv2d(in_channels=self.planes * (len(self.sppnet.levels)), out_channels=self.planes, kernel_size=1, stride=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=self.planes * (len(self.sppnet.levels) + 1), out_channels=self.planes//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.planes//4),
            nn.ReLU(inplace=True)
        )
        self.final_out = nn.Conv2d(self.planes//4, self.out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        N, C, H, W = x.size()
        res_out = self.resnet(x)
        pool_out = self.sppnet(res_out)
        #pool_out = self.conv_pre(pool_out)
        #print(res_out.size())
        x = torch.cat((res_out, pool_out), 1)
        x = self.decoder(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x = self.final_out(x)
        return x


class DeepLabV3(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3, self).__init__()
        self.deeplab = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=True)
        self.final_out = nn.Conv2d(in_channels=21, out_channels=cfg.MODEL.NUM_SEGMENTS, kernel_size=1, stride=1)
    
    def forward(self, x):
        x = self.deeplab(x)['out']
        x = self.final_out(x)
        return x

'''
nets = {}
nets['BasicParseNet'] = BasicParseNet(res_path=path, pretrained=True)
nets['Resnet101'] = BaseResnet(res_path=path)
nets['DeepLabV3'] = DeepLabV3()
#nets['PreTrainModel'] = PreTrainModel(name=name1, weights_path=path1, pretrained=True)

t1 = torch.randn(2, 3, 473, 473)
for name in nets.keys():
    net = nets[name].eval()
    start = time.time()
    out = net(t1)
    end = time.time()
    print(out.shape)
    print('{} time: {}'.format(name, end - start))
'''