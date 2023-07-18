import math
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import torch
from torch.nn.modules.utils import _triple, _pair, _single
import softpool_cuda

__all__ = ['Res2Net', 'res2net50']

model_urls = {
    'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth',
}

class Bottle2neck (nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super (Bottle2neck, self).__init__ ()

        width = int (math.floor (planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d (inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d (width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = SoftPool2d(kernel_size=3, stride=stride)
        convs = []
        bns = []
        for i in range (self.nums):
            convs.append (nn.Conv2d (width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append (nn.BatchNorm2d (width))
        self.convs = nn.ModuleList (convs)
        self.bns = nn.ModuleList (bns)

        self.conv3 = nn.Conv2d (width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d (planes * self.expansion)

        self.relu = nn.ReLU (inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1 (x)
        out = self.bn1 (out)
        out = self.relu (out)

        spx = torch.split (out, self.width, 1)
        for i in range (self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i] (sp)
            sp = self.relu (self.bns[i] (sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat ((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat ((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat ((out, self.pool (spx[self.nums])), 1)

        out = self.conv3 (out)
        out = self.bn3 (out)

        if self.downsample is not None:
            residual = self.downsample (x)

        out += residual
        out = self.relu (out)

        return out

class Res2Net (nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super (Res2Net, self).__init__ ()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d (3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d (64)
        self.relu = nn.ReLU (inplace=True)
        self.maxpool = nn.MaxPool2d (kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer (block, 64, layers[0])
        self.layer2 = self._make_layer (block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer (block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer (block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d (1)
        self.fc = nn.Linear (512 * block.expansion, num_classes)

        for m in self.modules ():
            if isinstance (m, nn.Conv2d):
                nn.init.kaiming_normal_ (m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance (m, nn.BatchNorm2d):
                nn.init.constant_ (m.weight, 1)
                nn.init.constant_ (m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential (
                nn.Conv2d (self.inplanes, planes * block.expansion,
                           kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d (planes * block.expansion),
            )

        layers = []
        layers.append (block (self.inplanes, planes, stride, downsample=downsample,
                              stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range (1, blocks):
            layers.append (block (self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential (*layers)

    def forward(self, x):
        x = self.conv1 (x)
        x = self.bn1 (x)
        x = self.relu (x)
        x = self.maxpool (x)

        x = self.layer1 (x)
        x = self.layer2 (x)
        x = self.layer3 (x)
        x = self.layer4 (x)

        x = self.avgpool (x)
        x = x.view (x.size (0), -1)
        x = self.fc (x)

        return x


def res2net50(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net (Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict (model_zoo.load_url (model_urls['res2net50_26w_4s']))
    return model


def res2net50_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net (Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict (model_zoo.load_url (model_urls['res2net50_26w_4s']))
    return model


def res2net101_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net (Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict (model_zoo.load_url (model_urls['res2net101_26w_4s']))
    return model


def res2net50_26w_6s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net (Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=6, **kwargs)
    if pretrained:
        model.load_state_dict (model_zoo.load_url (model_urls['res2net50_26w_6s']))
    return model


def res2net50_26w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net (Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=8, **kwargs)
    if pretrained:
        model.load_state_dict (model_zoo.load_url (model_urls['res2net50_26w_8s']))
    return model


def res2net50_48w_2s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_48w_2s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net (Bottle2neck, [3, 4, 6, 3], baseWidth=48, scale=2, **kwargs)
    if pretrained:
        model.load_state_dict (model_zoo.load_url (model_urls['res2net50_48w_2s']))
    return model


def res2net50_14w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_14w_8s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net (Bottle2neck, [3, 4, 6, 3], baseWidth=14, scale=8, **kwargs)
    if pretrained:
        model.load_state_dict (model_zoo.load_url (model_urls['res2net50_14w_8s']))
    return model

class CUDA_SOFTPOOL1d(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None):
        # Create contiguous tensor (if tensor is not contiguous)
        no_batch = False
        if len(input.size()) == 2:
            no_batch = True
            input.unsqueeze_(0)
        B, C, D = input.size()
        kernel = _single(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _single(stride)
        oD = (D-kernel[0]) // stride[0] + 1
        output = input.new_zeros((B, C, oD))
        softpool_cuda.forward_1d(input.contiguous(), kernel, stride, output)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        # Create contiguous tensor (if tensor is not contiguous)
        grad_input = torch.zeros_like(ctx.saved_tensors[0])
        saved = [grad_output.contiguous()] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride] + [grad_input]
        softpool_cuda.backward_1d(*saved)
        # Gradient underflow
        saved[-1][torch.isnan(saved[-1])] = 0
        return saved[-1], None, None

class CUDA_SOFTPOOL2d(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None):
        # Create contiguous tensor (if tensor is not contiguous)
        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, H, W = input.size()
        kernel = _pair(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _pair(stride)
        oH = (H - kernel[0]) // stride[0] + 1
        oW = (W - kernel[1]) // stride[1] + 1
        output = input.new_zeros((B, C, oH, oW))
        softpool_cuda.forward_2d(input.contiguous(), kernel, stride, output)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        # Create contiguous tensor (if tensor is not contiguous)
        grad_input = torch.zeros_like(ctx.saved_tensors[0])
        saved = [grad_output.contiguous()] + list(ctx.saved_tensors) + [ctx.kernel,ctx.stride] + [grad_input]
        softpool_cuda.backward_2d(*saved)
        # Gradient underflow
        saved[-1][torch.isnan(saved[-1])] = 0
        return saved[-1], None, None

class CUDA_SOFTPOOL3d(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None):
        # Create contiguous tensor (if tensor is not contiguous)
        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, D, H, W = input.size()
        kernel = _triple(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _triple(stride)
        oD = (D - kernel[0]) // stride[0] + 1
        oH = (H - kernel[1]) // stride[1] + 1
        oW = (W - kernel[2]) // stride[2] + 1
        output = input.new_zeros((B, C, oD, oH, oW))
        softpool_cuda.forward_3d(input.contiguous(), kernel, stride, output)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        # Create contiguous tensor (if tensor is not contiguous)
        grad_input = torch.zeros_like(ctx.saved_tensors[0])
        saved = [grad_output.contiguous()] + list(ctx.saved_tensors) + [ctx.kernel,ctx.stride] + [grad_input]
        softpool_cuda.backward_3d(*saved)
        # Gradient underflow
        saved[-1][torch.isnan(saved[-1])] = 0
        return saved[-1], None, None

def soft_pool1d(x, kernel_size=2, stride=None, force_inplace=False):
    if x.is_cuda and not force_inplace:
        x = CUDA_SOFTPOOL1d.apply(x, kernel_size, stride)
        # Replace `NaN's if found
        if torch.isnan(x).any():
            return torch.nan_to_num(x)
        return x
    kernel_size = _single(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _single(stride)
    # Get input sizes
    _, c, d = x.size()
    # Create exponential mask (should be similar to max-like pooling)
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    e_x = torch.clamp(e_x , float(0), float('inf'))
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x d] -> [b x c x d']
    x = F.avg_pool1d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool1d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))
    return torch.clamp(x , float(0), float('inf'))

def soft_pool2d(x, kernel_size=2, stride=None, force_inplace=False):
    if x.is_cuda and not force_inplace:
        x = CUDA_SOFTPOOL2d.apply(x, kernel_size, stride)
        # Replace `NaN's if found
        if torch.isnan(x).any():
            return torch.nan_to_num(x)
        return x
    kernel_size = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _pair(stride)
    # Get input sizes
    _, c, h, w = x.size()
    # Create exponential mask (should be similar to max-like pooling)
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    e_x = torch.clamp(e_x , float(0), float('inf'))
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x d] -> [b x c x d']
    x = F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))
    return torch.clamp(x , float(0), float('inf'))

def soft_pool3d(x, kernel_size=2, stride=None, force_inplace=False):
    if x.is_cuda and not force_inplace:
        x = CUDA_SOFTPOOL3d.apply(x, kernel_size, stride)
        # Replace `NaN's if found
        if torch.isnan(x).any():
            return torch.nan_to_num(x)
        return x
    kernel_size = _triple(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _triple(stride)
    # Get input sizes
    _, c, d, h, w = x.size()
    # Create exponential mask (should be similar to max-like pooling)
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    e_x = torch.clamp(e_x , float(0), float('inf'))
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x d x h x w] -> [b x c x d' x h' x w']
    x = F.avg_pool3d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool3d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))
    return torch.clamp(x , float(0), float('inf'))

class SoftPool1d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None, force_inplace=False):
        super(SoftPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.force_inplace = force_inplace

    def forward(self, x):
        return soft_pool1d(x, kernel_size=self.kernel_size, stride=self.stride, force_inplace=self.force_inplace)

class SoftPool2d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None, force_inplace=False):
        super(SoftPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.force_inplace = force_inplace

    def forward(self, x):
        return soft_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, force_inplace=self.force_inplace)

class SoftPool3d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None, force_inplace=False):
        super(SoftPool3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.force_inplace = force_inplace

    def forward(self, x):
        return soft_pool3d(x, kernel_size=self.kernel_size, stride=self.stride, force_inplace=self.force_inplace)

if __name__ == '__main__':
    images = torch.rand (1, 3, 224, 224).cuda (0)
    model = res2net50_26w_4s(pretrained=True)
    model = model.cuda (0)
    print (model (images).size ())