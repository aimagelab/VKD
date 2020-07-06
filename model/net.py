import torch
from torch import nn
from torchvision.models.densenet import densenet121

from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet34
from torchvision.models.resnet import resnet50
from torchvision.models.resnet import resnet101

from model.cbam.resnet_bam import resnet50_bam
from model.cbam.resnet_bam import resnet101_bam

from argparse import Namespace
from itertools import chain

import inspect


class Backbone(nn.Module):

    RESNET_18 = 'resnet18'
    RESNET_34 = 'resnet34'
    RESNET_50 = 'resnet50'
    RESNET_101 = 'resnet101'
    RESNET_50_BAM = 'resnet50bam'
    RESNET_101_BAM = 'resnet101bam'
    DENSENET = 'densenet121'
    MOBILENET = 'mobilenet'

    def __init__(self, btype: str, pretrained: bool = True, last_stride: int = 1):

        super(Backbone, self).__init__()

        assert btype in [Backbone.RESNET_18, Backbone.RESNET_34, Backbone.RESNET_50,
                         Backbone.RESNET_50_BAM, Backbone.RESNET_101, Backbone.RESNET_101_BAM,
                         Backbone.DENSENET, Backbone.MOBILENET]

        if btype in [Backbone.RESNET_18, Backbone.RESNET_34, Backbone.RESNET_50,
                     Backbone.RESNET_101, Backbone.RESNET_50_BAM, Backbone.RESNET_101_BAM]:
            self.features_layers, self.output_shape = \
                self.get_resnet_backbone_layers(btype, pretrained, last_stride)

        if btype in [Backbone.DENSENET]:
            self.features_layers, self.output_shape = self.get_denset_backbone_layers(pretrained)

        if btype in [Backbone.MOBILENET]:
            self.features_layers, self.output_shape = self.get_mobilenet_backbone_layers(pretrained, last_stride)

        self.btype = btype
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def get_net(btype: str, pretrained: bool):
        if btype == Backbone.RESNET_18:
            return resnet18(pretrained=pretrained, zero_init_residual=True), 512
        elif btype == Backbone.RESNET_34:
            return resnet34(pretrained=pretrained, zero_init_residual=True), 512
        elif btype == Backbone.RESNET_50:
            return resnet50(pretrained=pretrained, zero_init_residual=True), 2048
        elif btype == Backbone.RESNET_101:
            return resnet101(pretrained=pretrained, zero_init_residual=True), 2048
        elif btype == Backbone.RESNET_50_BAM:
            return resnet50_bam(pretrained=pretrained, zero_init_residual=True), 2048
        elif btype == Backbone.RESNET_101_BAM:
            return resnet101_bam(pretrained=pretrained, zero_init_residual=True), 2048
        elif btype == Backbone.DENSENET:
            return densenet121(pretrained=pretrained), 1024
        elif btype == Backbone.MOBILENET:
            model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2',
                                   pretrained=pretrained), 1280
            return model
        raise ValueError()

    @staticmethod
    def get_mobilenet_backbone_layers(pretrained: bool, last_stride: int):
        mobilenet, num_out_channels = Backbone.get_net(Backbone.MOBILENET, pretrained=pretrained)
        mobilenet_features = mobilenet.features
        mobilenet_features[14].conv[1][0].stride = (last_stride, last_stride)
        mobilenet_features[18][2] = nn.Sequential()
        return nn.ModuleList(mobilenet_features.children()), num_out_channels

    @staticmethod
    def get_resnet_backbone_layers(btype: str, pretrained: bool, last_stride: int):

        assert last_stride in [1, 2]

        resnet, output_shape = Backbone.get_net(btype, pretrained)

        if btype in [Backbone.RESNET_18, Backbone.RESNET_34]:
            resnet.layer4[0].conv1.stride = (last_stride, last_stride)
        if btype in [Backbone.RESNET_50, Backbone.RESNET_101,
                     Backbone.RESNET_50_BAM, Backbone.RESNET_101_BAM]:
            resnet.layer4[0].conv2.stride = (last_stride, last_stride)

        resnet.layer4[0].downsample[0].stride = (last_stride, last_stride)

        resnet.layer4[-1].relu = nn.Sequential()  # replace relu with empty sequential

        resnet_layers = [
            nn.Sequential(
                resnet.conv1, resnet.bn1,
                resnet.relu,
                resnet.maxpool),
            nn.Sequential(resnet.layer1),
            nn.Sequential(resnet.layer2),
            nn.Sequential(resnet.layer3),
            nn.Sequential(resnet.layer4),
        ]

        return nn.ModuleList(resnet_layers), output_shape

    @staticmethod
    def get_denset_backbone_layers(pretrained: bool):
        dnet = densenet121(pretrained=pretrained)
        original_model = list(dnet.children())[0]
        return nn.ModuleList(original_model.children()), 1024

    def get_output_shape(self):
        return self.output_shape

    def backbone_features(self, x: torch.Tensor):
        for m in self.features_layers:
            x = m(x)
        return x

    def forward(self, x: torch.Tensor):
        b, v, c, h, w = x.shape
        x = x.reshape(b * v, c, h, w)
        x = self.backbone_features(x)
        x = self.avgpool(x)
        return x

    def __call__(self, *args, **kwargs):
        return super(Backbone, self).__call__(*args, **kwargs)


BACKBONES = [Backbone.RESNET_18, Backbone.RESNET_34, Backbone.RESNET_50,
             Backbone.RESNET_101, Backbone.RESNET_50_BAM, Backbone.RESNET_101_BAM,
             Backbone.DENSENET, Backbone.MOBILENET]


class ClassificationLayer(nn.Module):

    def __init__(self, num_classes, feat_in: int):
        super(ClassificationLayer, self).__init__()

        self.feat_in = feat_in
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(self.feat_in)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.feat_in, self.num_classes, bias=False)

        self.bottleneck.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def forward(self, feats):
        return self.classifier(self.bottleneck(feats))

    @staticmethod
    def weights_init_kaiming(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def weights_init_classifier(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias:
                nn.init.constant_(m.bias, 0.0)

    def __call__(self, *args, **kwargs):
        return super(ClassificationLayer, self).__call__(*args, **kwargs)


class TriNet(nn.Module):

    def __init__(self, backbone_type: str, num_classes: int, pretrained: bool):

        super(TriNet, self).__init__()

        _, _, _, values = inspect.getargvalues(inspect.currentframe())
        self.hparams = {key: values[key] for key in values.keys()
                        if key not in ('self', '__class__')}

        self.backbone = Backbone(btype=backbone_type, pretrained=pretrained)

        self.aggregator = MeanAggregator()

        self.classifier = ClassificationLayer(num_classes=num_classes,
                                              feat_in=self.backbone.get_output_shape())

    def get_hparams(self):
        return self.hparams

    def backbone_features(self, x: torch.Tensor):
        b, v, c, h, w = x.shape
        x = self.backbone(x)
        out_shape = [b, v, self.backbone.output_shape]
        x = x.reshape(*out_shape)
        return x

    def forward(self, x: torch.Tensor, return_logits: bool = False):

        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        b, v, c, h, w = x.shape
        x = self.backbone(x)  # out before BN

        x_agg = self.aggregator(x.view(b, v, self.backbone.output_shape))

        if return_logits:
            # Note: this applies mean AFTER classifier, not before.
            x_class = self.classifier(x.view(b * v, self.backbone.output_shape))
            x_class = x_class.view(b, v, self.classifier.num_classes)
            x_class = torch.mean(x_class, dim=1)
            return x_agg, x_class

        return x_agg

    def teacher_mode(self):
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                m.track_running_stats = False
        self.train()

    def student_mode(self):
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                m.track_running_stats = True
        self.train()

    def reinit_layers(self, reinitl4: bool, reinitl3: bool):

        if reinitl4 or reinitl3:
            self.classifier.load_state_dict(
                ClassificationLayer(self.classifier.num_classes,
                                    self.classifier.feat_in).state_dict())

        r, _ = Backbone.get_net(self.backbone.btype, True)

        if reinitl4:
            if self.backbone.btype == Backbone.DENSENET:
                block = self.backbone.features_layers[-2]
                block.load_state_dict(r._modules['features'].denseblock4.state_dict())
            elif self.backbone.btype == Backbone.MOBILENET:
                pass
            else:
                block = self.backbone.features_layers[-1][0]
                block.load_state_dict(r.layer4.state_dict())

        if reinitl3:
            raise ValueError()

    def block_parameters(self, reinitl4: bool, reinitl3: bool):
        first_idx = []
        if reinitl4: first_idx.append(4)
        if reinitl3: first_idx.append(3)
        base_params = [ list(f.parameters()) for i, f in enumerate(self.backbone.features_layers)
                             if i not in first_idx ]
        upper_params = [ list(f.parameters()) for i, f in enumerate(self.backbone.features_layers)
                             if i in first_idx ]
        upper_params.append(list(self.classifier.parameters()))
        base_params = chain(*base_params)
        upper_params = chain(*upper_params)
        return base_params, upper_params

    def __call__(self, *args, **kwargs):
        return super(TriNet, self).__call__(*args, **kwargs)


def get_model(args: Namespace, num_pids: int):
    return TriNet(backbone_type=args.backbone, pretrained=args.pretrained,
                  num_classes=num_pids)


class MeanAggregator(nn.Module):

    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, x: torch.Tensor):
        return x.mean(dim=1)

    def __call__(self, *args, **kwargs):
        return super(MeanAggregator, self).__call__(*args, **kwargs)
