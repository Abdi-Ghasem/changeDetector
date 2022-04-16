# Original Author       : Ghasem Abdi, ghasem.abdi@yahoo.com
# File Last Update Date : April 15, 2022

import torch
import torch.nn as nn

from timm.models.layers import create_attn
from ..utils import noop, aggregate_features

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        bn = nn.BatchNorm2d(num_features=out_channels)
        relu = nn.ReLU(inplace=True)
        super(ConvLayer, self).__init__(conv, bn, relu)

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        conv1 = ConvLayer(in_channels=in_channels, out_channels=out_channels)
        conv2 = ConvLayer(in_channels=out_channels, out_channels=out_channels)
        super(CenterBlock, self).__init__(conv1, conv2)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, attention_type=None):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels=in_channels + skip_channels, out_channels=out_channels)
        self.conv2 = ConvLayer(in_channels=out_channels, out_channels=out_channels)

        self.attention1 = create_attn(attn_type=attention_type, channels=in_channels + skip_channels) \
            if attention_type else noop
        self.attention2 = create_attn(attn_type=attention_type, channels=out_channels) \
            if attention_type else noop

    def forward(self, x, skip=None):
        x = nn.functional.interpolate(input=x, scale_factor=2, mode='nearest')
        
        if skip is not None:
            x = torch.cat(tensors=[x, skip], dim=1)
            x = self.attention1(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class UNetPlusPlusDecoder(nn.Module):
    """"create a UNet decoder network for change detection:
    encoder_channel: a List of integers which specify in_channels parameter for convolutions used in encoder,
    decoder_channel: a List of integers which specify in_channels parameter for convolutions used in decoder,
    classes: an integer number of output classes/channels,
    center: a boolean indicator of using center block, default is False,
    fusion_type: a string name of the feature fusion type (avilable options are 'sum', 'diff', and 'concat') \
        for aggregating features of different spatial resolution, default is 'concat',
    attention_type: a string name of the attentiona module (that is part of the timm library), default is None"""

    def __init__(
        self, 
        encoder_channels, 
        decoder_channels, 
        classes, 
        center=False, 
        fusion_type='concat', 
        attention_type=None
    ):
        super(UNetPlusPlusDecoder, self).__init__()

        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels

        self.fusion_type = fusion_type
        if self.fusion_type == 'concat':
            self.in_channels[0] *= 2
            self.skip_channels = [chl * 2 for chl in self.skip_channels]
        
        self.center = CenterBlock(in_channels=head_channels, out_channels=head_channels) \
            if center else noop

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                out_ch = self.out_channels[layer_idx]
                skip_ch = self.out_channels[layer_idx] * (layer_idx - depth_idx) + self.skip_channels[layer_idx]
                in_ch = self.skip_channels[layer_idx - 1] \
                    if depth_idx == layer_idx and depth_idx != 0 else self.in_channels[layer_idx]

                blocks[f'x_{depth_idx}_{layer_idx}'] = \
                    DecoderBlock(in_channels=in_ch, skip_channels=skip_ch, out_channels=out_ch, attention_type=attention_type)
        blocks[f'x_{0}_{len(self.in_channels) - 1}'] = \
            DecoderBlock(in_channels=self.in_channels[-1], skip_channels=0, out_channels=self.out_channels[-1], attention_type=attention_type)
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

        self.segmentation_head = nn.Conv2d(in_channels=(decoder_channels[(-1)]), out_channels=classes, kernel_size=3, padding=1)
        
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(tensor=m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(tensor=m.weight)
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)

    def forward(self, features):
        features = aggregate_features(x1=features[0], x2=features[1], fusion_type=self.fusion_type)
        features = features[::-1]

        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f'x_{depth_idx}_{depth_idx}'](features[depth_idx], features[depth_idx + 1])
                    dense_x[f'x_{depth_idx}_{depth_idx}'] = output

                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f'x_{idx}_{dense_l_i}'] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f'x_{depth_idx}_{dense_l_i}'] = \
                        self.blocks[f'x_{depth_idx}_{dense_l_i}'](dense_x[f'x_{depth_idx}_{dense_l_i - 1}'], cat_features)
        
        x = self.blocks[f'x_{0}_{self.depth}'](dense_x[f'x_{0}_{self.depth - 1}'])
        
        x = self.segmentation_head(x)
        return x