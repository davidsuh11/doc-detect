"""
Everything related to building model

Author: David Suh 
Website: david-suh.pages.dev
Email: suhdavid11 (at) gmail (dot) com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

WEIGHT = "COCO_WITH_VOC_LABELS_V1"
"""
Builds the model for training and evaluation. A DeepLabV3 model for fine-tuning is used.
"""
def build_model(backbone='mobilnetv3', pretrained=True):
    if backbone == 'mobilenetv3':
        model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=WEIGHT if pretrained else None)
    elif backbone == 'resnet50':
        model = models.segmentation.deeplabv3_resnet50(weights=WEIGHT if pretrained else None)
    elif backbone == 'resnet101':
        model = models.segmentation.deeplabv3_resnet101(weights=WEIGHT if pretrained else None)
    else:
        raise ValueError(f'Invalid backbone: {backbone}')

    # Replace the final layer of the head with a new one
    model.classifier[4] = nn.Conv2d(model.classifier[4].in_channels, 2, kernel_size=(1, 1), stride=(1, 1))
    return model

