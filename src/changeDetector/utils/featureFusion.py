# Original Author       : Ghasem Abdi, ghasem.abdi@yahoo.com
# File Last Update Date : April 15, 2022

import torch
import torch.nn as nn

class feature_fusion(nn.Module):
    def __init__(self, fusion_type='concat'):
        super(feature_fusion, self).__init__()
        self.fusion_type = fusion_type

    def forward(self, x1, x2):
        options = {
            'sum':lambda x1, x2: torch.add(input=x1, other=x2), 
            'diff':lambda x1, x2: torch.sub(input=x1, other=x2).abs(), 
            'concat':lambda x1, x2: torch.cat(tensors=[x1, x2], dim=1)
        }
        return options[self.fusion_type](x1, x2)

def aggregate_features(x1, x2, fusion_type='concat'):
    return [feature_fusion(fusion_type=fusion_type)(x1[i], x2[i]) for i in range(len(x1))]