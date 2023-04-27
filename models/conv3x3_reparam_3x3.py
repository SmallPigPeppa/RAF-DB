import torch.nn as nn
import torch
import torch.nn.init as init


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Conv3x3_Reparam(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, groups=1, dilation=1):
        super(Conv3x3_Reparam, self).__init__()
        self.conv3x3 = conv3x3(in_planes, out_planes, stride=stride, groups=groups, dilation=dilation)
        self.branch3x3 = conv3x3(in_planes, out_planes, stride=stride, groups=groups, dilation=dilation)
        self.use_branch = False
        # self.zero_branch()

    def forward(self, x):
        z1 = self.conv3x3(x)
        z2 = self.branch3x3(x)
        if self.use_branch:
            return z1 + z2
        else:
            return z1

    def fix_conv(self):
        self.conv3x3.weight.requires_grad = False

    def fix_branch(self):
        self.branch3x3.weight.requires_grad =False

    def set_branch(self, use_branch=True):
        self.use_branch = use_branch

    def re_param(self):
        with torch.no_grad():
            self.conv3x3.weight.data = self.get_equivalent_kernel_bias()
            self.zero_branch()

    def get_equivalent_kernel_bias(self):
        return self.conv3x3.weight + self.branch3x3.weight

    # def pad_1x1_to_3x3(self, kernel1x1):
    #     return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def zero_branch(self):
        init.zeros_(self.branch3x3.weight)



if __name__ == '__main__':
    x = torch.rand([4, 3, 32, 32])
    m = Conv3x3_Reparam(in_planes=3, out_planes=6)
    m.use_branch = True
    # m.fix_conv()
    # m.fix_branch()
    # m.zero_branch()
    # m.eval()
    y = m(x)
    print(y[0][0][0])
    m.re_param()

    y2 = m(x)
    print(y2[0][0][0])
