# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.models.sub_modules.cbam import BasicBlock as CbamBasicBlock

class ResizeNet(nn.Module):
    """
    将输入特征转换为指定尺寸
    """
    def __init__(self, input_dim, out_dim, unify_method = 'conv'):
        super().__init__()
        """
        target_size: C, H, W
        input_dim: channel of source feature
        output_dim: channel of target feature
        method: channel align method
        """
        
        # self.channel_unify = nn.Sequential(
        #     nn.Conv2d(input_dim, input_dim//raito, kernel_size=3,
        #               stride=1, padding=1),
        #     nn.BatchNorm2d(input_dim//raito, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        
        #     nn.Conv2d(input_dim//raito, input_dim, kernel_size=3,
        #               stride=1, padding=1),
        #     nn.BatchNorm2d(input_dim, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(out_dim, eps=1e-3,
        #                    momentum=0.01),
        #     nn.ReLU()
            
        # )
        
        if unify_method == 'conv':
            self.channel_unify = nn.Sequential(
                nn.Conv2d(input_dim, out_dim, kernel_size=1), 
            )
        elif unify_method == 'conv3':
            self.channel_unify = nn.Sequential(
                nn.Conv2d(input_dim, out_dim, kernel_size=3), 
            )
        elif unify_method == 'cbam':
            downsample = nn.Sequential(nn.Conv2d(input_dim, out_dim, stride=1, kernel_size=1), 
                                       nn.ReLU(nn.ReLU(inplace=True)))
            self.channel_unify = CbamBasicBlock(input_dim, out_dim, downsample=downsample)
        
        
    def forward(self, x, target_size):
        _, C, H, W = x.size()
        
        if (H, W) != target_size[1:]: 
            x = F.interpolate(x, size=target_size[1:], mode='bilinear')
        
        if C != target_size[0]:    
            x = self.channel_unify(x)
        
        return x
        
