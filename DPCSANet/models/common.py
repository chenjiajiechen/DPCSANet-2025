# This file contains modules common to various models

import math
import sys
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import warnings
# sys.path.append('../')  # to run '$ python *.py' files in subdirectories

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, xyxy2xywh
from utils.plots import color_list
# from einops import rearrange


class CABlock(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(CABlock, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.ca = ChannelAttention(c1)

    def forward(self, x):
        x1 = self.ca(x) * x
        return self.conv(x1 + x)

    def fuseforward(self, x):
        return self.act(self.conv(x))

# class CABlock(nn.Module):
#     # Standard convolution
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(CABlock, self).__init__()
#         self.conv1 = nn.Conv2d(c1, c2//2, k, s, autopad(k, p), groups=g, bias=False)
#         self.conv2 = nn.Conv2d(c1, c2//2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn1 = nn.BatchNorm2d(c2//2)
#         self.bn2 = nn.BatchNorm2d(c2//2)
#         self.act1 = nn.Hardswish() if act else nn.Identity()
#         self.act2 = nn.Hardswish() if act else nn.Identity()
#         self.ca = ChannelAttention(c2//2)
#
#     def forward(self, x):
#         x1 = self.act1(self.bn1(self.conv1(x)))
#         x1 = self.ca(x1) * x1
#         x2 = self.act2(self.bn2(self.conv1(x)))
#         return torch.cat((x1, x2), dim=1)
#
#     def fuseforward(self, x):
#         return self.act(self.conv(x))

# 标准卷积层 + CBAM
class CBAM(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(CBAM, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.ca = ChannelAttention(c2)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


# add CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CA(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CA, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)  # [Batch, C, W+H, 1]
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h


# -------------------------------------------------------------------------------------------------------------
def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GnConv(nn.Module):
    def __init__(self, dim, order=4, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, (1, 1))

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        # else:
        #     self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, (1, 1))

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], (1, 1)) for i in range(order - 1)]
        )

        self.scale = s

        # print('[gconv]', order, '阶与维度=', self.dims, '尺度=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        # B, C, H, W = x.shape
        fused_x = self.proj_in(x)
        # print("self.dims:", self.dims)
        # print("sum(self.dims):", sum(self.dims))
        # print("fused_x.shape:", fused_x.shape)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]  # 逐元素乘法，在PyTorch中，逐元素乘法使用*操作符，而矩阵乘法通常使用torch.matmul()或者@操作符

        return self.proj_out(x)


# class C3GnConv(nn.Module):
#     # CSP Bottleneck with 3 convolutions
#     def __init__(self, c1, c2, n=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super(C3GnConv, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
#         self.m = nn.Sequential(*[GnConv(c_) for _ in range(n)])
#         # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
#
#     def forward(self, x):
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class GnBlock(nn.Module):
    def __init__(self, dim, shortcut=False, layer_scale_init_value=1e-6):
        super().__init__()
        self.shortcut = shortcut
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = GnConv(dim, order=5)
        self.norm2 = LayerNorm(dim, eps=1e-6, data_format='channels_last')
        self.pwconv1 = nn.Linear(dim, 2 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(1, C, 1, 1)
        else:
            gamma1 = 1
        x = (x + gamma1 * self.gnconv(self.norm1(x))) if self.shortcut else gamma1 * self.gnconv(self.norm1(x))
        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        if self.gamma2 is not None:
            gamma2 = self.gamma2.view(1, C, 1, 1)
            x = gamma2 * x
        else:
            gamma2 = 1
        x = (input + x) if self.shortcut else x
        return x
# -------------------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------------
# 自注意机制
# -------------------------------------------------------------------------------------------------------------
class ExtraPositionPromptSABottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, n_dims, size, shortcut=True):  # ch_in, ch_out, shortcut, groups, expansion
        super(ExtraPositionPromptSABottleneck, self).__init__()

        height = size
        width = size
        self.cv1 = Conv(n_dims, n_dims//2, 1, 1)
        self.cv2 = Conv(n_dims//2, n_dims, 1, 1)
        '''MHSA PARAGRAMS'''
        self.query = nn.Conv2d(n_dims//2, n_dims//2, kernel_size=(1, 1))
        self.key = nn.Conv2d(n_dims//2, n_dims//2, kernel_size=(1, 1))
        self.value = nn.Conv2d(n_dims//2, n_dims//2, kernel_size=(1, 1))
        self.extra = nn.Conv2d(n_dims//2, n_dims//2, kernel_size=(1, 1))

        # self.extra_p1 = nn.Parameter(torch.randn([1, n_dims//2, height, 1]), requires_grad=True)
        # self.extra_p2 = nn.Parameter(torch.randn([1, n_dims//2, 1, width]), requires_grad=True)

        self.rel_h = nn.Parameter(torch.randn([1, n_dims//2, height, 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims//2, 1, width]), requires_grad=True)


        self.softmax = nn.Softmax(dim=-1)
        self.add = shortcut

    def forward(self, x):
        x1=self.cv1(x)
        n_batch, C, width, height = x1.size()
        q = self.query(x1).view(n_batch, C, -1)
        k = self.key(x1).view(n_batch, C, -1)
        v = self.value(x1).view(n_batch, C, -1)
        e = self.extra(x1).view(n_batch, C, -1)

        content_content = torch.bmm(q.permute(0, 2, 1), k)  # 全局联系

        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)

        # If you want to use resolution-agnostic positional encoding, you can uncomment the following lines.
        # See details in https://github.com/WindVChen/DRENet/issues/10.
        # Note that the performance of this resolution-agnostic positional encoding is not tested.
        # content_position = (self.rel_h + self.rel_w)
        # content_position = nn.functional.interpolate(content_position, (int(content_content.shape[-1]**0.5), int(content_content.shape[-1]**0.5)), mode='bilinear')
        # content_position = content_position.view(1, C, -1).permute(0, 2, 1)

        content_position = torch.matmul(content_position, e)
        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)

        return x + self.cv2(out) if self.add else self.cv2(out)


class BottleneckResAtnMHSA(nn.Module):
    # Standard bottleneck
    def __init__(self, n_dims, size, shortcut=True):  # ch_in, ch_out, shortcut, groups, expansion
        super(BottleneckResAtnMHSA, self).__init__()

        height=size
        width=size
        self.cv1 = Conv(n_dims, n_dims//2, 1, 1)
        self.cv2 = Conv(n_dims//2, n_dims, 1, 1)
        '''MHSA PARAGRAMS'''
        self.query = nn.Conv2d(n_dims//2, n_dims//2, kernel_size=1)
        self.key = nn.Conv2d(n_dims//2, n_dims//2, kernel_size=1)
        self.value = nn.Conv2d(n_dims//2, n_dims//2, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, n_dims//2, height, 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims//2, 1, width]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)
        self.add = shortcut

    def forward(self, x):
        x1=self.cv1(x)
        n_batch, C, width, height = x1.size()
        q = self.query(x1).view(n_batch, C, -1)
        k = self.key(x1).view(n_batch, C, -1)
        v = self.value(x1).view(n_batch, C, -1)

        content_content = torch.bmm(q.permute(0, 2, 1), k)  # 全局联系

        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)

        # If you want to use resolution-agnostic positional encoding, you can uncomment the following lines.
        # See details in https://github.com/WindVChen/DRENet/issues/10.
        # Note that the performance of this resolution-agnostic positional encoding is not tested.
        # content_position = (self.rel_h + self.rel_w)
        # content_position = nn.functional.interpolate(content_position, (int(content_content.shape[-1]**0.5), int(content_content.shape[-1]**0.5)), mode='bilinear')
        # content_position = content_position.view(1, C, -1).permute(0, 2, 1)

        content_position = torch.matmul(content_position, q)
        # energy = content_content
        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)


        return x + self.cv2(out) if self.add else self.cv2(out)


class ExtraAddBottleneckResAtnMHSA(nn.Module):
    # Standard bottleneck
    def __init__(self, n_dims, size, shortcut=True):  # ch_in, ch_out, shortcut, groups, expansion
        super(ExtraAddBottleneckResAtnMHSA, self).__init__()

        height=size
        width=size
        self.cv1 = Conv(n_dims, n_dims//2, 1, 1)
        self.cv2 = Conv(n_dims//2, n_dims, 1, 1)
        '''MHSA PARAGRAMS'''
        self.query = nn.Conv2d(n_dims//2, n_dims//2, kernel_size=1)
        self.key = nn.Conv2d(n_dims//2, n_dims//2, kernel_size=1)
        self.value = nn.Conv2d(n_dims//2, n_dims//2, kernel_size=1)
        self.extra = nn.Conv2d(n_dims//2, n_dims//2, kernel_size=(1, 1))

        self.rel_h = nn.Parameter(torch.randn([1, n_dims//2, height, 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims//2, 1, width]), requires_grad=True)
        self.extra_p1 = nn.Parameter(torch.randn([1, n_dims//2, height, 1]), requires_grad=True)
        self.extra_p2 = nn.Parameter(torch.randn([1, n_dims//2, 1, width]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        self.add = shortcut

    def forward(self, x):
        x1=self.cv1(x)
        n_batch, C, width, height = x1.size()
        q = self.query(x1).view(n_batch, C, -1)
        k = self.key(x1).view(n_batch, C, -1)
        v = self.value(x1).view(n_batch, C, -1)
        e = self.extra(x1).view(n_batch, C, -1)

        content_content = torch.bmm(q.permute(0, 2, 1), k)  # 全局联系

        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        extra_p = (self.extra_p1 + self.extra_p2).view(1, C, -1).permute(0, 2, 1)

        content_position = torch.matmul(content_position, q)
        extra_content = torch.matmul(extra_p, e)

        energy = content_content + content_position + extra_content
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)


        return x + self.cv2(out) if self.add else self.cv2(out)


class ExtraMHSA(nn.Module):
    # Standard bottleneck
    def __init__(self, n_dims, size, shortcut=True):  # ch_in, ch_out, shortcut, groups, expansion
        super(ExtraMHSA, self).__init__()

        height = size
        width = size
        self.cv1 = Conv(n_dims, n_dims // 2, 1, 1)
        self.cv2 = Conv(n_dims // 2, n_dims, 1, 1)
        self.cv3 = Conv(n_dims // 2, n_dims, 1, 1)
        '''MHSA PARAGRAMS'''
        self.query = nn.Conv2d(n_dims // 2, n_dims // 2, kernel_size=1)
        self.key = nn.Conv2d(n_dims // 2, n_dims // 2, kernel_size=1)
        self.value = nn.Conv2d(n_dims // 2, n_dims // 2, kernel_size=1)

        self.position = nn.Conv2d(n_dims // 2, n_dims // 2, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, n_dims // 2, height, 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims // 2, 1, width]), requires_grad=True)

        # self.extra_f = nn.Parameter(torch.randn([1, n_dims // 2, height, width]), requires_grad=True)
        self.extra_f_h = nn.Parameter(torch.randn([1, n_dims // 2, height, 1]), requires_grad=True)
        self.extra_f_w = nn.Parameter(torch.randn([1, n_dims // 2, 1, width]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)
        self.add = shortcut

    def forward(self, x):
        x1 = self.cv1(x)
        n_batch, C, width, height = x1.size()
        q = self.query(x1).view(n_batch, C, -1)
        k = self.key(x1).view(n_batch, C, -1)
        v = self.value(x1).view(n_batch, C, -1)
        p = self.position(x1).view(n_batch, C, -1)
        # extra_f = self.extra_f.view(n_batch, C, -1)

        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        extra_f = (self.extra_f_h + self.extra_f_w).view(n_batch, C, -1)


        content_content = torch.bmm(q.permute(0, 2, 1), k)  # 全局联系
        content_position = torch.matmul(content_position, q)
        attention = self.softmax(content_content + content_position)

        out1 = torch.bmm(v, attention.permute(0, 2, 1))
        out1 = out1.view(n_batch, C, width, height)

        content_extra = torch.bmm(p.permute(0, 2, 1), extra_f)
        content_extra_energy = self.softmax(content_extra)
        out2 = torch.bmm(extra_f, content_extra_energy.permute(0, 2, 1))
        out2 = out2.view(n_batch, C, width, height)

        return x + self.cv2(out1) + self.cv3(out2) if self.add else self.cv2(out1)


class ConvAttentionAdd(nn.Module):
    # Standard bottleneck
    def __init__(self, n_dims, size, shortcut=True):  # ch_in, ch_out, shortcut, groups, expansion
        super(ConvAttentionAdd, self).__init__()
        height = size
        width = size
        self.cv1 = Conv(n_dims, n_dims//2, 1, 1)
        self.cv2 = Conv(n_dims//2, n_dims, 1, 1)
        '''MHSA PARAGRAMS'''
        self.query = nn.Conv2d(n_dims//2, n_dims//2, kernel_size=1)
        self.key = nn.Conv2d(n_dims//2, n_dims//2, kernel_size=1)
        self.value = nn.Conv2d(n_dims//2, n_dims//2, kernel_size=1)
        self.v_conv = Conv_1(n_dims//2)
        # self.v_dconv = DConv(n_dims//2)

        self.rel_h = nn.Parameter(torch.randn([1, n_dims//2, height, 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims//2, 1, width]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)
        self.add = shortcut
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x1 = self.cv1(x)
        n_batch, C, width, height = x1.size()
        q = self.query(x1).view(n_batch, C, -1)
        k = self.key(x1).view(n_batch, C, -1)
        v = self.value(x1).view(n_batch, C, -1)
        v_conv = self.v_conv(x1).view(n_batch, C, width, height)
        # v_dconv = self.v_dconv(x1).view(n_batch, C, width, height)

        # v_o = self.value(x1).view(n_batch, C, -1)
        # v = v_o + self.v_conv(x1).view(n_batch, C, -1)
        # v = v_o + self.v_conv(x1).view(n_batch, C, -1) + self.v_dconv(x1).view(n_batch, C, -1)

        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        content_content = torch.bmm(q.permute(0, 2, 1), k)  # 全局联系
        content_position = torch.matmul(content_position, q)
        energy = content_content + content_position
        attention = self.softmax(energy)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)
        # v = v.view(n_batch, C, width, height)
        return x + self.cv2(out + self.gamma * v_conv)
        # return x + self.cv2(self.gamma * out + v) if self.add else self.cv2(out)


class C3ResAtnMHSA(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, size=14, shortcut=True, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3ResAtnMHSA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[ConvAttentionAdd(c_, size, shortcut=True) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


# class Attention(nn.Module):
#     def __init__(self, in_dim, in_feature, out_feature):
#         super(Attention, self).__init__()
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
#         self.query_line = nn.Linear(in_features=in_feature, out_features=out_feature)
#         self.key_line = nn.Linear(in_features=in_feature, out_features=out_feature)
#         self.s_conv = nn.Conv2d(in_channels=1, out_channels=in_dim, kernel_size=1)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         q = rearrange(self.query_line(rearrange(self.query_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b h 1')
#         k = rearrange(self.key_line(rearrange(self.key_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b 1 h')
#         att = rearrange(torch.matmul(q, k), 'b h w -> b 1 h w')
#         att = self.softmax(self.s_conv(att))
#         return att

def conv_relu_bn(in_channel, out_channel, dirate):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=dirate,
                  dilation=dirate),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


# class Conv_1(nn.Module):
#     def __init__(self, in_dim):
#         super(Conv_1, self).__init__()
#         self.convs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, 1) for _ in range(1)])
#
#     def forward(self, x):
#         for conv in self.convs:
#             x = conv(x)
#         return x
class Conv_1(nn.Module):
    def __init__(self, in_dim):
        super(Conv_1, self).__init__()
        self.convs = conv_relu_bn(in_dim, in_dim, 1)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

# class DConv(nn.Module):
#     def __init__(self, in_dim):
#         super(DConv, self).__init__()
#         dilation = [2, 4, 2]
#         self.dconvs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, dirate) for dirate in dilation])
#
#     def forward(self, x):
#         for dconv in self.dconvs:
#             x = dconv(x)
#         return x


class DConv(nn.Module):
    def __init__(self, in_dim):
        super(DConv, self).__init__()
        self.dconvs = conv_relu_bn(in_dim, in_dim, 2)

    def forward(self, x):
        for dconv in self.dconvs:
            x = dconv(x)
        return x


# class ConvAttention(nn.Module):
#     def __init__(self, in_dim, in_feature, out_feature):
#         super(ConvAttention, self).__init__()
#         self.conv = Conv_1(in_dim)
#         self.dconv = DConv(in_dim)
#         self.att = Attention(in_dim, in_feature, out_feature)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         q = self.conv(x)
#         k = self.dconv(x)
#         v = q + k
#         att = self.att(x)
#         out = torch.matmul(att, v)
#         return self.gamma * out + v + x





def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)  # math.gcd计算最大公约数


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, bias=False):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class inv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, bias=False):  # ch_in, ch_out, kernel, stride, padding, groups
        super(inv, self).__init__()
        self.INV = False
        self.inChannel = c1
        if self.inChannel<4 or self.inChannel<16 or not self.INV:
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=bias)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        else:
            kernel_size = k
            stride = s
            channels = c1
            channelsOut = c2
            self.kernel_size = k
            self.stride = s
            self.channels = c1
            reduction_ratio = 4
            self.group_channels = 16
            self.groups = self.channels // self.group_channels
            self.conv1 = nn.Sequential(nn.Conv2d(channels, channels // reduction_ratio, 1, groups=g, bias=bias),
                                       nn.BatchNorm2d(channels // reduction_ratio),
                                       nn.ReLU())
            self.conv2 = nn.Conv2d(channels // reduction_ratio, kernel_size ** 2 * self.groups, 1, groups=g, bias=bias)
            if stride > 1:
                self.avgpool = nn.AvgPool2d(stride, stride)
            self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)
            self.bn = nn.BatchNorm2d(channels)
            self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
            self.conv3 = nn.Conv2d(channels, channelsOut, 1, groups=g, bias=bias)
            self.bn2 = nn.BatchNorm2d(channelsOut)
            self.act2 = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        if self.inChannel < 4 or self.inChannel < 16 or not self.INV:
            return self.act(self.bn(self.conv(x)))
        else:
            weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
            b, c, h, w = weight.shape
            weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
            out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
            out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
            out = self.act(self.bn(out))
            return self.act2(self.bn2(self.conv3(out)))

    def fuseforward(self, x):
        if self.inChannel < 4 or self.inChannel < 16 or not self.INV:
            return self.act(self.conv(x))
        else:
            return self.act2(self.conv3(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP

    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class HHSPP(nn.Module):
    def __init__(self, c1, c2):
        super(HHSPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        # self.cv = Conv(c1, c2//2, 1, 1)
        self.cv1 = Conv(c1, c_, 1, 1)     # 这里对应第一个CBL
        self.cv2 = Conv(c_ * 5, c2, 1, 1)   # 这里对应SPP操作里的最后一个CBL

        self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.m2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.m3 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.m4 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
        # self.k_spp = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # out1 = self.cv(x)
        y0 = self.cv1(x)
        y1 = self.m1(y0)
        y2 = self.m2(y1)
        y3 = self.m3(y1)
        y4 = self.m4(y1)
        out = self.cv2(torch.cat([y0, y1, y2, y3, y4], 1))
        return out
        # return out + self.k_spp * x

        # return torch.cat((out1, out2), dim=1)
# class HHSPP(nn.Module):
#     def __init__(self, c1, c2):
#         super(HHSPP, self).__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)  # First CBL
#         # self.cv2 = Conv(c_ * 5, c2, 1, 1)  # Last CBL in SPP
#         self.cv2 = Conv(c_ * 5, c2, 1, 1)  # Last CBL in SPP
#
#         self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
#         self.m2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
#         self.m3 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
#         self.m4 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
#
#         self.ca = ChannelAttention(c_ * 5)  # Channel Attention for concatenated output
#
#     def forward(self, x):
#         y0 = self.cv1(x)
#         y1 = self.m1(y0)
#         y2 = self.m2(y1)
#         y3 = self.m3(y1)
#         y4 = self.m4(y1)
#         concatenated = torch.cat([y0, y1, y2, y3, y4], 1)
#         # concatenated = torch.cat([y0, y1, y3, y4], 1)
#         weighted = self.ca(concatenated) * concatenated
#         out = self.cv2(weighted)
#         return out  # Add residual connection


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class HSPP(nn.Module):
    def __init__(self, c1, c2):
        super(HSPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)  # 这里对应第一个CBL
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # 这里对应SPP操作里的最后一个CBL
        self.m1 = nn.AvgPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.m2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.m3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m1(x)
        y2 = self.m2(y1)
        y3 = self.m3(y1)
        out = self.cv2(torch.cat([x, y1, y2, y3], 1))
        return out


class pureSPP(nn.Module):
    def __init__(self, c1, c2):
        super(pureSPP, self).__init__()
        # self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.m2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.m3 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.m4 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

    def forward(self, x):
        # y0 = self.m1(x)
        y1 = self.m2(x)
        y2 = self.m3(x)
        y3 = self.m4(x)
        return torch.cat([x, y1, y2, y3], 1)


class HGSPP(nn.Module):
    def __init__(self, c1, c2):
        super(HGSPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        # self.cv = Conv(c1, c2//2, 1, 1)
        self.cv1 = Conv(c1, c_, 1, 1)     # 这里对应第一个CBL
        self.cv2 = Conv(c_ * 5, c2, 1, 1)   # 这里对应SPP操作里的最后一个CBL
        # self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        # self.cv3 = Conv(c2, c2, 1, 1)
        self.m1 = nn.AvgPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.m2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.m3 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.m4 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

    def forward(self, x):
        # out1 = self.cv(x)
        y0 = self.cv1(x)
        y1 = self.m1(y0)
        y2 = self.m2(y1)
        y3 = self.m3(y1)
        y4 = self.m4(y1)
        out = self.cv2(torch.cat([y0, y1, y2, y3, y4], 1))
        return out
        # return torch.cat((out1, out2), dim=1)


class HHSPP5913(nn.Module):
    def __init__(self, c1, c2):
        super(HHSPP5913, self).__init__()
        c_ = c1 // 2  # hidden channels
        # self.cv = Conv(c1, c2//2, 1, 1)
        self.cv1 = Conv(c1, c_, 1, 1)     # 这里对应第一个CBL
        self.cv2 = Conv(c_ * 4, c2, 1, 1)   # 这里对应SPP操作里的最后一个CBL
        # self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        # self.cv3 = Conv(c2, c2, 1, 1)
        self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.m3 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.m4 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

    def forward(self, x):
        # out1 = self.cv(x)
        y0 = self.cv1(x)
        y1 = self.m1(y0)
        y3 = self.m3(y1)
        y4 = self.m4(y1)
        out = self.cv2(torch.cat([y0, y1, y3, y4], 1))
        return out
        # return torch.cat((out1, out2), dim=1)

class HHSPP559(nn.Module):
    def __init__(self, c1, c2):
        super(HHSPP559, self).__init__()
        c_ = c1 // 2  # hidden channels
        # self.cv = Conv(c1, c2//2, 1, 1)
        self.cv1 = Conv(c1, c_, 1, 1)     # 这里对应第一个CBL
        self.cv2 = Conv(c_ * 4, c2, 1, 1)   # 这里对应SPP操作里的最后一个CBL
        # self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        # self.cv3 = Conv(c2, c2, 1, 1)
        self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.m2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.m3 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)

    def forward(self, x):
        # out1 = self.cv(x)
        y0 = self.cv1(x)
        y1 = self.m1(y0)
        y2 = self.m2(y1)
        y3 = self.m3(y1)
        out = self.cv2(torch.cat([y0, y1, y2, y3], 1))
        return out
        # return torch.cat((out1, out2), dim=1)

class HHSPP55(nn.Module):
    def __init__(self, c1, c2):
        super(HHSPP55, self).__init__()
        c_ = c1 // 2  # hidden channels
        # self.cv = Conv(c1, c2//2, 1, 1)
        self.cv1 = Conv(c1, c_, 1, 1)     # 这里对应第一个CBL
        self.cv2 = Conv(c_ * 3, c2, 1, 1)   # 这里对应SPP操作里的最后一个CBL
        # self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        # self.cv3 = Conv(c2, c2, 1, 1)
        self.m1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.m2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)

    def forward(self, x):
        # out1 = self.cv(x)
        y0 = self.cv1(x)
        y1 = self.m1(y0)
        y2 = self.m2(y1)
        out = self.cv2(torch.cat([y0, y1, y2], 1))
        return out
        # return torch.cat((out1, out2), dim=1)

class HySPP(nn.Module):
    def __init__(self, c1, c2):
        super(HySPP, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 5, c2, 1, 1)
        self.avg1 = nn.AvgPool2d(kernel_size=5, stride=1, padding=5 // 2)
        # self.avg2 = nn.AvgPool2d(kernel_size=9, stride=1, padding=9 // 2)
        # self.avg3 = nn.AvgPool2d(kernel_size=13, stride=1, padding=5 // 2)

        self.max1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=3 // 2)
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=3 // 2)
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=3 // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.avg1(x)
        y2 = self.max1(x)

        # y3 = self.avg2(y2)
        y4 = self.max2(y2)

        # y5 = self.avg3(y4)
        y6 = self.max3(y4)

        return self.cv2(torch.cat([x, y1, y2, y4, y6], dim=1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1, selectPos=None):
        super(Concat, self).__init__()
        self.d = dimension
        self.p=selectPos

    def forward(self, x):
        if isinstance(self.p, int):
            return torch.cat([x[0][self.p],x[1]], self.d)
        else:
            return torch.cat(x, self.d)


class ConcatFusionFactor(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(ConcatFusionFactor, self).__init__()
        self.d = dimension
        self.factor=torch.nn.Parameter(torch.FloatTensor([1]))

    def forward(self, x):
        x[0]=x[0]*self.factor
        return torch.cat(x, self.d)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    img_size = 640  # inference size (pixels)
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=720, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(720,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(720,1280,3)
        #   numpy:           = np.zeros((720,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,720,1280)  # BCHW
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1 = [], []  # image and inference shapes
        for i, im in enumerate(imgs):
            if isinstance(im, str):  # filename or uri
                im = Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)  # open
            im = np.array(im)  # to numpy
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32

        # Inference
        with torch.no_grad():
            y = self.model(x, augment, profile)[0]  # forward
        y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS

        # Post-process
        for i in range(n):
            scale_coords(shape1, y[i][:, :4], shape0[i])

        return Detections(imgs, y, self.names)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, names=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)

    def display(self, pprint=False, show=False, save=False, render=False):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'Image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f'{n} {self.names[int(c)]}s, '  # add to string
                if show or save or render:
                    img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        # str += '%s %.2f, ' % (names[int(cls)], conf)  # label
                        ImageDraw.Draw(img).rectangle(box, width=4, outline=colors[int(cls) % 10])  # plot
            if pprint:
                print(str)
            if show:
                img.show(f'Image {i}')  # show
            if save:
                f = f'results{i}.jpg'
                str += f"saved to '{f}'"
                img.save(f)  # save
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results

    def show(self):
        self.display(show=True)  # show results

    def save(self):
        self.display(save=True)  # save results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def __len__(self):
        return self.n

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


class LocalReconstruct(nn.Module):
    def __init__(self, c1, c2):
        super(LocalReconstruct, self).__init__()
        self.reconstruct = nn.Sequential(
            Conv(c1, c2, 1, 1),
            Conv(c2, c2//4, 3, 1),
            Conv(c2//4, c2, 1, 1)
        )

    def forward(self, x):
        x0=self.reconstruct(x[0])
        x1=self.reconstruct(x[1])
        x2=self.reconstruct(x[2])
        return x0,x1,x2


class SEAtn(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEAtn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class AtnMut(nn.Module):
    def __init__(self, start, end):
        super(AtnMut, self).__init__()
        self.start=start
        self.end=end

    def forward(self, x):
        atn= x[0][:, self.start:self.end, :, :]
        obj= x[1]
        out= obj*atn.expand_as(obj)
        return out


class MHSA(nn.Module):
    def __init__(self, n_dims, size):
        super(MHSA, self).__init__()

        height = size
        width = size
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, n_dims, height, 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims, 1, width]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, C, -1)
        k = self.key(x).view(n_batch, C, -1)
        v = self.value(x).view(n_batch, C, -1)

        content_content = torch.bmm(q.permute(0, 2, 1), k)

        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)

        return out


## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, c1, conv=Conv):
        super(RCAN, self).__init__()

        n_resgroups = 1
        n_resblocks = 1
        n_feats = c1
        kernel_size = 3
        reduction = 16
        scale = 2
        act = nn.SiLU()

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]

        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        res = self.body(x)
        res += x

        x = self.tail(res)

        return x


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        for _ in range(int(math.log(scale, 2))):
            m.append(conv(n_feat, 4 * n_feat, 3, bias = bias))
            m.append(nn.PixelShuffle(2))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())

        super(Upsampler, self).__init__(*m)