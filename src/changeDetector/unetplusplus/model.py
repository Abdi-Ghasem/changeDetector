# Original Author       : Ghasem Abdi, ghasem.abdi@yahoo.com
# File Last Update Date : April 15, 2022

import timm
import torch.nn as nn

from .decoder import UNetPlusPlusDecoder
from typing import Optional, List

class UNetPlusPlus(nn.Module):
  """"create a UNet++ fully convolutional neural network for change detection:
  in_channels: an integer number of input channels for the model, default is 3,
  encoder_name: a string name of the encoder (that is part of the timm library) for extracting features of different spatial resolution, default is 'resnet34',
  pretrained: a boolean indicator of using pretrained weight (True) or random initialization (False) for the encoder, default is True,
  decoder_channel: a List of integers which specify in_channels parameter for convolutions used in decoder, default is (256, 128, 64, 32, 16),
  encoder_fusion_type: a string name of the feature fusion type (avilable options are 'sum', 'diff', and 'concat') \
    for aggregating features of different spatial resolution, default is 'concat',
  decoder_attention_type: a string name of the attentiona module (that is part of the timm library), default is None,
  classes: an integer number of output classes/channels"""

  def __init__(
    self, 
    in_channels: int = 3, 
    encoder_name: str = 'resnet34', 
    pretrained: Optional[bool] = True, 
    decoder_channels: List[int] = (256, 128, 64, 32, 16), 
    encoder_fusion_type: str = 'concat', 
    decoder_attention_type: Optional[str] = None, 
    classes: int = 1
  ):
    super(UNetPlusPlus, self).__init__()
    
    self.encoder = timm.create_model(
      model_name=encoder_name,
      pretrained=pretrained,
      in_chans=in_channels,
      features_only=True
    )

    self.decoder = UNetPlusPlusDecoder(
      encoder_channels=self.encoder.feature_info.channels()[::-1],
      decoder_channels=decoder_channels,
      classes=classes,
      center=True if encoder_name.startswith('vgg') else False,
      fusion_type=encoder_fusion_type,
      attention_type=decoder_attention_type
    )

  def forward(self, x1, x2):
    features = (self.encoder(x1), self.encoder(x2))
    out = self.decoder(features)
    return out