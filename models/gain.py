import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, target=None):
        n, c, h, w = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_map = self.layer4(x)

        x = self.avgpool(feature_map)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        
        if target is None:
            gcam = (self.linear.weight[x.argmax(-1)].unsqueeze(-1).unsqueeze(-1) * feature_map).sum(dim=1, keepdim=True)
        else:
            gcam = (self.linear.weight[target].unsqueeze(-1).unsqueeze(-1) * feature_map).sum(dim=1, keepdim=True)
        gcam = F.interpolate(gcam, (h, w), mode='bilinear', align_corners=True)
        gcam_min = gcam.min()
        gcam_max = gcam.max()
        scaled_gcam = (gcam - gcam_min) / (gcam_max - gcam_min)

        return x, scaled_gcam


def resnet(model_name, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model_dict = {
        'resnet18': [BasicBlock, [2, 2, 2, 2]], 
        'resnet34': [BasicBlock, [3, 4, 6, 3]], 
        'resnet50': [Bottleneck, [3, 4, 6, 3]], 
        'resnet101': [Bottleneck, [3, 4, 23, 3]], 
        'resnet152': [Bottleneck, [3, 8, 36, 3]], 
    }
    model = ResNet(model_dict[model_name][0], model_dict[model_name][1], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls[model_name])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


class SoftMask(nn.Module):
    def __init__(self, scale=100, threshold=0.5):
        super().__init__()
        self.scale = scale
        self.threshold = threshold
        
    def forward(self, x):
        return F.sigmoid(-self.scale * (x - self.threshold))
    
    
class GAIN(nn.Module):
    def __init__(self, num_classes, model_name='resnet50', pretrained=True, **kwargs):
        super().__init__()
        self.model = resnet(model_name, pretrained, num_classes=num_classes, **kwargs)
        self.softmask = SoftMask()
    
    def forward(self, x, target=None):
        out, gcam = self.model(x, target)
        mask = self.softmask(gcam)
        x_mean = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        x_masked = x * (1 - mask) + x_mean * mask
        out_masked, gcam_masked = self.model(x_masked, target)
        
        return out, out_masked, gcam
    
    
class GAINLoss(nn.Module):
    def __init__(self, alpha=1, omega=10):
        super().__init__()
        self.alpha = alpha
        self.omega = omega
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_mining = nn.NLLLoss()
        self.criterion_seg = nn.MSELoss()
        
    def forward(self, out, out_masked, cam, target, mask=None):
        n, c = out.shape
        loss_cls = self.criterion_cls(out, target)
        
        target_one_hot = torch.zeros((n, c), dtype=torch.float32, device=target.device)
        target_one_hot.scatter_(1, target.view(-1, 1), 1)
        loss_mining = (F.sigmoid(out_masked) * target_one_hot).sum() / n
        
        if mask is not None:
            loss_seg = torch.zeros((1, ), dtype=torch.float32, device=loss_cls.device)
#             loss_seg = -(mask * torch.log(cam + 1e-6) + \
#                          (1 - mask) * torch.log(1 - cam + 1e-6)).mean()
            loss = loss_cls + self.alpha * loss_mining + self.omega * loss_seg
        else:
            loss_seg = torch.zeros((1, ), dtype=torch.float32, device=loss_cls.device)
            loss = loss_cls + self.alpha * loss_mining
        
        return loss, (loss_cls, loss_mining, loss_seg)