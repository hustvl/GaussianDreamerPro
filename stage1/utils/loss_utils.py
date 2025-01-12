#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]

def tv_loss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])  
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    return 2*(h_tv/count_h+w_tv/count_w)/batch_size    

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def cal_opacity_loss(_opacity, eps=1e-6):
    '''
    opacity \in [0, 1]
    '''
    opacity = _opacity * (1 - 2*eps) + eps
    return torch.mean(-opacity * torch.log2(opacity))
    # return torch.mean(opacity * torch.log2(opacity) + 0.5)

def cal_splat_loss(scales):
    # min_scales = torch.min(scales, dim=-1)[0]
    # return torch.mean(min_scales)
    return torch.mean(scales)

def cal_tv_loss(inputs, losstype='l1', stage=1):
    ''' 
    Returns TV norm for input values.
    inputs: [c, H, W]
    '''
    if losstype == 'pooling':
        kernel_size = 2 * stage + 1
        padding = stage
        with torch.no_grad():
            smoothed_normals = F.pad(inputs, (padding, padding, padding, padding), mode='replicate')
            smoothed_normals = F.avg_pool2d(smoothed_normals, kernel_size=kernel_size, stride=1)
        loss = (inputs - smoothed_normals).abs().mean()
        # loss = ((inputs - smoothed_normals)**2).mean()
    elif losstype == 'l2' or losstype == 'l1':
        step = stage
        v00 = inputs[:, :-step, :-step]
        v01 = inputs[:, :-step, step:]
        v10 = inputs[:, step:, :-step]
        if losstype == 'l2':
            loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2).mean()
        elif losstype == 'l1':
            loss = (torch.abs(v00 - v01) + torch.abs(v00 - v10)).mean()
        else:
            raise ValueError('Not supported losstype.')
    return loss

class NormalLoss:
    def __init__(self, H, W, focal_x, focal_y, device='cuda'):
        cx, cy = 0.5 * W - 0.5, 0.5 * H - 0.5
        Y, X = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing='ij')
        view_dirs = torch.stack([(X-cx)/focal_x, (Y-cy)/focal_y, torch.ones_like(X)], dim=0).float().to(device)
        self.view_dirs = view_dirs / torch.norm(view_dirs, dim=0, keepdim=True) # [3, H, W]

    def grad_operator(self, depth, radius=1): # distance: [1, H, W]
        points = depth * self.view_dirs # [3, H, W]
        points_pad = F.pad(points, (radius, radius, radius, radius), "replicate")
        stride = int(radius * 2)
        grad_x = points_pad[:, :, stride:] - points_pad[:, :, :-stride]
        grad_y = points_pad[:, stride:] - points_pad[:, :-stride]
        depth_normal = torch.cross(grad_x[:, radius:-radius], grad_y[:, :, radius:-radius], dim=0) # [3, H, W]
        depth_normal /= (torch.norm(depth_normal, dim=0, keepdim=True) + 1e-7)
        return depth_normal # [3, H, W]

    def __call__(self, _normals, _depth, accums, stage=1, accum_thres=0.95):
        '''
        normals: [3, H, W]
        depth: [1, H, W]
        '''
        normals = _normals.clone().permute(1,2,0) # [H, W, 3]
        depth = _depth.detach().clone()
        accum = accums.detach().permute(1,2,0)
        valid_mask = (accum[..., 0] > accum_thres)
        if valid_mask.sum() == 0:
            return torch.tensor(0.).to(normals.device)
        normals[valid_mask] /= accum[valid_mask]
        with torch.no_grad():
            depth_normal = self.grad_operator(depth, radius=stage).permute(1,2,0) # [H, W, 3]
        loss = (1. - torch.sum((normals * depth_normal)[valid_mask], dim=-1).abs()).mean()
        return loss
