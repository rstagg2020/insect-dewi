import torch
import torch.nn as nn
import torch.nn.functional as F
from models.projector import Projector
from models.resnet_utils import conv1x1, BasicBlock, Bottleneck


class TransformerNeck(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, d_model, 1, 1))  # Learnable, will interpolate


class CosineClassifier(nn.Module):
    def __init__(self, in_features, num_classes, init_scale=20.0):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = nn.Parameter(torch.tensor([init_scale]))
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, w) * self.scale


class DeWi(nn.Module):
    def __init__(self, block, layers, num_classes=102, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(DeWi, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
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
        
        # projectors
        self.projector1 = Projector(input_dim=1024, projection_dims=[2048] + [4096] * 3)
        self.projector2 = Projector(input_dim=2048, projection_dims=[2048] + [4096] * 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4, inplace=False)
        self.fc = nn.Linear(8192, num_classes)

        # Transformer neck
        self.neck = None

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

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        p1 = self.projector1(x)
        x = self.layer4(x)

        # Adaptive Transformer Neck
        if self.neck is not None:
            B, C, H, W = x.shape
            seq = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            pos_embed = F.interpolate(self.neck.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            seq = seq + pos_embed
            transformer_out = self.neck.transformer(seq)  # [B, H*W, C]
            # Proper spatial residual: reshape back to [B, C, H, W] before adding
            x_spatial = transformer_out.permute(0, 2, 1).reshape(B, C, H, W)
            x = x + x_spatial

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        p2 = self.projector2(x)
        p = torch.cat((p1, p2), dim=1)
        embedding = self.dropout(p)
        logits = self.fc(embedding)
        return p, logits

    def forward(self, x):
        return self._forward_impl(x)


def _dewi(block, layers, pretrained, pth_url, **kwargs):
    model = DeWi(block, layers, **kwargs)
    if pretrained:
        pretrained_dict = torch.hub.load_state_dict_from_url(pth_url)
        del pretrained_dict['fc.weight']
        del pretrained_dict['fc.bias']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    # Add Transformer Neck
    model.neck = TransformerNeck(d_model=2048)
    return model


def dewi_resnet50(pth_url=None, pretrained=False, **kwargs):
    return _dewi(Bottleneck, [3, 4, 6, 3], pretrained, pth_url, **kwargs)


def dewi_resnet101(pth_url=None, pretrained=False, **kwargs):
    return _dewi(Bottleneck, [3, 4, 23, 3], pretrained, pth_url, **kwargs)


def dewi_resnet152(pth_url=None, pretrained=False, **kwargs):
    return _dewi(Bottleneck, [3, 8, 36, 3], pretrained, pth_url, **kwargs)


def dewi_resnext50_32x4d(pth_url=None, pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _dewi(Bottleneck, [3, 4, 6, 3], pretrained, pth_url, **kwargs)


def dewi_resnext101_32x8d(pth_url=None, pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _dewi(Bottleneck, [3, 4, 23, 3], pretrained, pth_url, **kwargs)


def dewi_resnext101_64x4d(pth_url=None, pretrained=False, **kwargs):
    kwargs['groups'] = 64
    kwargs['width_per_group'] = 4
    return _dewi(Bottleneck, [3, 4, 23, 3], pretrained, pth_url, **kwargs)


def dewi_wide_resnet50_2(pth_url=None, pretrained=False, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _dewi(Bottleneck, [3, 4, 6, 3], pretrained, pth_url, **kwargs)


def dewi_wide_resnet101_2(pth_url=None, pretrained=False, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _dewi(Bottleneck, [3, 4, 23, 3], pretrained, pth_url, **kwargs)