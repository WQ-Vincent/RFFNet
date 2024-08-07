import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


import torchvision.models as models


class VGGLoss(nn.Module):
    def __init__(self, layers=[1,4,8]):
        super(VGGLoss, self).__init__()
        self.layers = layers # 'relu1_2', 'relu2_2', 'relu3_3'
        #self.content_layers = [11, 12] # 'relu4_2', 'relu4_3'
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).cuda()
        self.features = nn.Sequential(*list(vgg.features.children())[:max(self.layers)+1])
        self.criterion = nn.L1Loss()  # 使用L1损失函数

    def forward(self, pred, gt):
        pred_features = []
        gt_features = []
        x = pred
        y = gt
        for i, module in enumerate(self.features):
            x, y = module(x), module(y)
            if i in self.layers:
                pred_features.append(x)
                gt_features.append(y)
        loss = 0.0
        for pred_feat, gt_feat in zip(pred_features, gt_features):
            a = self.criterion(pred_feat, gt_feat)
            loss += a
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(PSNRLoss, self).__init__()
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1
        self.balance = 1.1

    def forward(self, inputs, targets):
        n, c, h, w = inputs.size()

        input_flat=inputs.view(-1)
        target_flat=targets.view(-1)

        intersecion=input_flat * target_flat
        unionsection=input_flat.pow(2).sum() + target_flat.pow(2).sum() + self.smooth
        loss=unionsection/(2 * intersecion.sum() + self.smooth)
        loss=loss.sum()

        return loss



