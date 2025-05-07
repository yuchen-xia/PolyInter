import torch
import torch.nn as nn
import numpy as np

class Focuser(nn.Module):
    """
    选择出特征的关键部分, 返回一个mask: [B, 1, H, W]
    """
    def __init__(self, args):
        super(Focuser, self).__init__()
        self.estimator = nn.Conv2d( \
                args["dim"], \
                args['anchor_number'], \
                kernel_size=1)
        
        self.threshold = args['threshold']
        
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            # Help filter out the outliers and introduce some context.
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(
            self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()
    
    def forward(self, x):
        """
        x: [B, C, H, W]
        mask: [B, 1, H, W]
        significant_feature: [B, anchor_num, H, W]
        """
        psm = self.estimator(x)
        
        # feature_show(psm[0], '/home/scz/HEAL/analysis/psm', type = 'mean')

        # 是否存在任何一种类型的目标
        mask, _ = psm.sigmoid().max(dim=1, keepdim = True)
        
        # feature_show(mask[0], '/home/scz/HEAL/analysis/mask0', type = 'mean')
        
        if self.smooth:
            mask = self.gaussian_filter(mask)
        
        # feature_show(mask[0], '/home/scz/HEAL/analysis/mask', type = 'mean')
        
        ones_mask = torch.ones_like(mask).to(x.device)
        zeros_mask = torch.zeros_like(mask).to(x.device)
        
        mask = torch.where(mask > self.threshold, ones_mask, zeros_mask)
        
        # 对cls选择出的结果进行监督
        return mask, psm