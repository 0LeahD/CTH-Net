from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import torch
from torch.nn.modules.utils import _triple, _pair, _single
class SoftPool2d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None, force_inplace=False):
        super(SoftPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.force_inplace = force_inplace
    def forward(self, x):
        return soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, force_inplace=self.force_inplace)
def soft_pool2d(x, kernel_size=2, stride=None, force_inplace=False):
    if x.is_cuda and not force_inplace:
        x = CUDA_SOFTPOOL2d.apply(x, kernel_size, stride)
        if torch.isnan(x).any():
            return torch.nan_to_num(x)
        return x
    kernel_size = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _pair(stride)
    _, c, h, w = x.size()
    e_x = torch.exp(x)
    return F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))
    
