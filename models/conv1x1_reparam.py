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


class Conv1x1_Reparam(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Conv1x1_Reparam, self).__init__()
        self.conv1x1 = conv1x1(in_planes, out_planes, stride=stride)
        self.branch1x1 = conv1x1(in_planes, out_planes, stride=stride)
        self.use_branch = False
        # self.zero_branch()

    def forward(self, x):
        z1 = self.conv1x1(x)
        z2 = self.branch1x1(x)
        if self.use_branch:
            return z1 + z2
        else:
            return z1

    def fix_conv(self):
        self.conv1x1.weight.requires_grad = False

    def fix_branch(self):
        self.branch1x1.weight.requires_grad =False



    def set_branch(self, use_branch=True):
        self.use_branch = use_branch

    def re_param(self):
        with torch.no_grad():
            self.conv1x1.weight.data = self.get_equivalent_kernel_bias()
            self.zero_branch()

    def get_equivalent_kernel_bias(self):
        return self.conv1x1.weight + self.branch1x1.weight

    def zero_branch(self):
        init.zeros_(self.branch1x1.weight)



if __name__ == '__main__':
    x = torch.rand([4, 3, 32, 32])
    m = Conv1x1_Reparam(in_planes=3, out_planes=6)
    m.use_branch = True
    m.fix_conv()
    m.fix_branch()
    # m.zero_branch()
    # m.eval()
    y = m(x)
    print(y[0][0][0])
    m.re_param()

    y2 = m(x)
    print(y2[0][0][0])
