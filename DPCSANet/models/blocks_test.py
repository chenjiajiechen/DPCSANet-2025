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


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


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

        self.rel_h = nn.Parameter(torch.randn([1, n_dims//2, height, 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims//2, 1, width]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)
        self.add = shortcut

    def forward(self, x):
        x1 = self.cv1(x)
        n_batch, C, width, height = x1.size()
        print(f"Shape of x1: {x1.shape}")  # Output the shape after the first convolution

        q = self.query(x1).view(n_batch, C, -1)
        k = self.key(x1).view(n_batch, C, -1)
        v = self.value(x1).view(n_batch, C, -1)
        e = self.extra(x1).view(n_batch, C, -1)
        print(f"Shapes of q, k, v, e: {q.shape}, {k.shape}, {v.shape}, {e.shape}")

        content_content = torch.bmm(q.permute(0, 2, 1), k)
        print(f"Shape of content_content: {content_content.shape}")

        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        print(f"Shape of content_position before matmul: {content_position.shape}")

        content_position = torch.matmul(content_position, e)
        print(f"Shape of content_position after matmul: {content_position.shape}")

        energy = content_content + content_position
        print(f"Shape of energy: {energy.shape}")

        attention = self.softmax(energy)
        print(f"Shape of attention: {attention.shape}")

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)
        print(f"Shape of out after bmm and view: {out.shape}")

        result = x + self.cv2(out) if self.add else self.cv2(out)
        print(f"Shape of result: {result.shape}")
        return result


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


class C3ResAtnMHSA(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, size=14, shortcut=True, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3ResAtnMHSA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[ExtraPositionPromptSABottleneck(c_, size, shortcut=True) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


if __name__=="__main__":
    # 创建一个随机输入张量
    x = torch.randn(1, 64, 32, 32)  # 假设输入张量大小为 [1, 64, 32, 32]

    # 创建BottleneckResAtnMHSA实例
    # bottleneck = BottleneckResAtnMHSA(n_dims=64, size=32, shortcut=True)
    # print("BottleneckResAtnMHSA输出大小:", bottleneck.size())

    # 创建CRMAConv实例
    crma_conv = C3ResAtnMHSA(c1=64, c2=32, n=1, size=32, shortcut=True, e=0.5)

    # 对输入张量进行前向传播
    # out_bottleneck, x = bottleneck(x)
    out_crma_conv = crma_conv(x)

    # 输出调整后的张量大小
    # print("调整后的BottleneckResAtnMHSA输出大小:", out_bottleneck.size())
    print("调整后的CRMAConv输出大小:", out_crma_conv.size())

    # 进行torch.cat操作
    # concatenated = torch.cat((out_bottleneck, out_crma_conv), dim=1)
    # print("成功进行torch.cat操作后的张量大小:", concatenated.size())