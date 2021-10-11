import torch
import torch.nn as nn
import math
import numpy as np
from torchvision import models
from torchvision import transforms

from mmdet.models.backbones.swin import SwinTransformer
from mmdet.models.backbones.resnet import ResNet

from Swin_Transformer.models.swin_transformer import SwinTransformer

class CustomResnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.model = models.resnet50(pretrained=True)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1. / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-stdv, stdv)
    
    def forward(self,x) :
        output = self.model(x)
        
        return output

class CustomSwinTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = SwinTransformer(depths=[2, 2, 18, 2]) #small

        num_ftrs = self.model.head.in_features
        self.model.head = nn.Linear(num_ftrs, num_classes)

        torch.nn.init.xavier_uniform_(self.model.head.weight)
        stdv = 1. / math.sqrt(self.model.head.weight.size(1))
        self.model.head.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        output = self.model(x)
        return output

'''
class CustomResnet50(nn.Module) :
    def __init__(self, num_classes):
        super().__init__()
        
        self.model   = ResNet(depth=50, init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(2048, num_classes)
        
        torch.nn.init.xavier_uniform_(self.fc.weight)
        stdv = 1. / math.sqrt(self.fc.weight.size(1))
        self.fc.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x) :
        x = self.model(x)
        output = x[3]
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.fc(output)
        return output
'''

if __name__ == "__main__" :
    #pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
    #model = SwinTransformer(init_cfg=dict(type='Pretrained', checkpoint=pretrained))
    #model = CustomResnet50(10)
    model = SwinTransformer(depths=[2, 2, 18, 2])
    print(model)
    #print(models.resnet50())
    x = torch.randn((1, 3, 224, 224))
    out = model(x)
    #print(CustomResnet50(1000))
    print(out.shape)