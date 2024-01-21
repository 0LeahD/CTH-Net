import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt
from ND_Crossentropy import CrossentropyND, TopKLoss

class FocalLoss (nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super (FocalLoss, self).__init__ ()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError ('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin (logit)
        num_class = logit.shape[1]

        if logit.dim () > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view (logit.size (0), logit.size (1), -1)
            logit = logit.permute (0, 2, 1).contiguous ()
            logit = logit.view (-1, logit.size (-1))
        target = torch.squeeze (target, 1)
        target = target.view (-1, 1)
        # print(logit.shape, target.shape)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones (num_class, 1)
        elif isinstance (alpha, (list, np.ndarray)):
            assert len (alpha) == num_class
            alpha = torch.FloatTensor (alpha).view (num_class, 1)
            alpha = alpha / alpha.sum ()
        elif isinstance (alpha, float):
            alpha = torch.ones (num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError ('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to (logit.device)

        idx = target.cpu ().long ()

        one_hot_key = torch.FloatTensor (target.size (0), num_class).zero_ ()
        one_hot_key = one_hot_key.scatter_ (1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to (logit.device)

        if self.smooth:
            one_hot_key = torch.clamp (
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum (1) + self.smooth
        logpt = pt.log ()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze (alpha)
        loss = -1 * alpha * torch.pow ((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean ()
        else:
            loss = loss.sum ()
        return loss


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss


class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        if debug:
            eroted, erosions = self.perform_erosion(
                pred.cpu().numpy(), target.cpu().numpy(), debug
            )
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(
                self.perform_erosion(pred.cpu().numpy(), target.cpu().numpy(), debug)
            ).float()

            loss = eroted.mean()

            return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        # print(inputs.shape, targets.shape) # (batch size, class_num, x,y,z), (batch size, 1, x,y,z)
        inputs, targets = self.prob_flatten(inputs, targets)
        # print(inputs.shape, targets.shape)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses

class CrossentropyND (torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """

    def forward(self, inp, target):
        target = target.long ()
        num_classes = inp.size ()[1]

        i0 = 1
        i1 = 2

        while i1 < len (inp.shape):  # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose (i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous ()
        inp = inp.view (-1, num_classes)

        target = target.view (-1, )

        return super (CrossentropyND, self).forward (inp, target)


class TopKLoss (CrossentropyND):
    """
    Network has to have NO LINEARITY!
    """

    def __init__(self, weight=None, ignore_index=-100, k=10):
        self.k = k
        super (TopKLoss, self).__init__ (weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        target = target[:, 0].long ()
        res = super (TopKLoss, self).forward (inp, target)
        num_voxels = np.prod (res.shape)
        res, _ = torch.topk (res.view ((-1,)), int (num_voxels * self.k / 100), sorted=False)
        return res.mean ()


class WeightedCrossEntropyLoss (torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """

    def __init__(self, weight=None):
        super (WeightedCrossEntropyLoss, self).__init__ ()
        self.weight = weight

    def forward(self, inp, target):
        target = target.long ()
        num_classes = inp.size ()[1]

        i0 = 1
        i1 = 2

        while i1 < len (inp.shape):  # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose (i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous ()
        inp = inp.view (-1, num_classes)

        target = target.view (-1, )
        wce_loss = torch.nn.CrossEntropyLoss (weight=self.weight)

        return wce_loss (inp, target)


class WeightedCrossEntropyLossV2 (torch.nn.Module):
    """
    WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    Network has to have NO LINEARITY!
    copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L121
    """

    def forward(self, net_output, gt):
        # compute weight
        # shp_x = net_output.shape
        # shp_y = gt.shape
        # print(shp_x, shp_y)
        # with torch.no_grad():
        #     if len(shp_x) != len(shp_y):
        #         gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        #     if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
        #         # if this is the case then gt is probably already a one hot encoding
        #         y_onehot = gt
        #     else:
        #         gt = gt.long()
        #         y_onehot = torch.zeros(shp_x)
        #         if net_output.device.type == "cuda":
        #             y_onehot = y_onehot.cuda(net_output.device.index)
        #         y_onehot.scatter_(1, gt, 1)
        # y_onehot = y_onehot.transpose(0,1).contiguous()
        # class_weights = (torch.einsum("cbxyz->c", y_onehot).type(torch.float32) + 1e-10)/torch.numel(y_onehot)
        # print('class_weights', class_weights)
        # class_weights = class_weights.view(-1)
        class_weights = torch.cuda.FloatTensor ([0.2, 0.8])
        gt = gt.long ()
        num_classes = net_output.size ()[1]
        # class_weights = self._class_weights(inp)

        i0 = 1
        i1 = 2

        while i1 < len (net_output.shape):  # this is ugly but torch only allows to transpose two axes at once
            net_output = net_output.transpose (i0, i1)
            i0 += 1
            i1 += 1

        net_output = net_output.contiguous ()
        net_output = net_output.view (-1, num_classes)  # shape=(vox_num, class_num)

        gt = gt.view (-1, )
        # print('*'*20)
        return F.cross_entropy (net_output, gt)  # , weight=class_weights

    # @staticmethod
    # def _class_weights(input):
    #     # normalize the input first
    #     input = F.softmax(input, _stacklevel=5)
    #     flattened = flatten(input)
    #     nominator = (1. - flattened).sum(-1)
    #     denominator = flattened.sum(-1)
    #     class_weights = Variable(nominator / denominator, requires_grad=False)
    #     return class_weights


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size (1)
    # new axis order
    axis_order = (1, 0) + tuple (range (2, tensor.dim ()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute (axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    transposed = transposed.contiguous ()
    return transposed.view (C, -1)


def compute_edts_forPenalizedLoss(GT):
    """
    GT.shape = (batch_size, x,y,z)
    only for binary segmentation
    """
    GT = np.squeeze (GT)
    res = np.zeros (GT.shape)
    for i in range (GT.shape[0]):
        posmask = GT[i]
        negmask = ~posmask
        pos_edt = distance_transform_edt (posmask)
        pos_edt = (np.max (pos_edt) - pos_edt) * posmask
        neg_edt = distance_transform_edt (negmask)
        neg_edt = (np.max (neg_edt) - neg_edt) * negmask
        res[i] = pos_edt / np.max (pos_edt) + neg_edt / np.max (neg_edt)
    return res


class DisPenalizedCE (torch.nn.Module):
    """
    Only for binary 3D segmentation

    Network has to have NO NONLINEARITY!
    """

    def forward(self, inp, target):
        # print(inp.shape, target.shape) # (batch, 2, xyz), (batch, 2, xyz)
        # compute distance map of ground truth
        with torch.no_grad ():
            dist = compute_edts_forPenalizedLoss (target.cpu ().numpy () > 0.5) + 1.0

        dist = torch.from_numpy (dist)
        if dist.device != inp.device:
            dist = dist.to (inp.device).type (torch.float32)
        dist = dist.view (-1, )

        target = target.long ()
        num_classes = inp.size ()[1]

        i0 = 1
        i1 = 2

        while i1 < len (inp.shape):  # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose (i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous ()
        inp = inp.view (-1, num_classes)
        log_sm = torch.nn.LogSoftmax (dim=1)
        inp_logs = log_sm (inp)

        target = target.view (-1, )
        # loss = nll_loss(inp_logs, target)
        loss = -inp_logs[range (target.shape[0]), target]
        # print(loss.type(), dist.type())
        weighted_loss = loss * dist

        return loss.mean ()


def nll_loss(input, target):
    """
    customized nll loss
    source: https://medium.com/@zhang_yang/understanding-cross-entropy-
    implementation-in-pytorch-softmax-log-softmax-nll-cross-entropy-416a2b200e34
    """
    loss = -input[range (target.shape[0]), target]
    return loss.mean ()

def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range (len (x.size ()))]
    rpt[1] = x.size (1)
    x_max = x.max (1, keepdim=True)[0].repeat (*rpt)
    e_x = torch.exp (x - x_max)
    return e_x / e_x.sum (1, keepdim=True).repeat (*rpt)


def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique (axes).astype (int)
    if keepdim:
        for ax in axes:
            inp = inp.sum (int (ax), keepdim=True)
    else:
        for ax in sorted (axes, reverse=True):
            inp = inp.sum (int (ax))
    return inp


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/dice_loss.py
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple (range (2, len (net_output.size ())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad ():
        if len (shp_x) != len (shp_y):
            gt = gt.view ((shp_y[0], 1, *shp_y[1:]))

        if all ([i == j for i, j in zip (net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long ()
            y_onehot = torch.zeros (shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda (net_output.device.index)
            y_onehot.scatter_ (1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack (tuple (x_i * mask[:, 0] for x_i in torch.unbind (tp, dim=1)), dim=1)
        fp = torch.stack (tuple (x_i * mask[:, 0] for x_i in torch.unbind (fp, dim=1)), dim=1)
        fn = torch.stack (tuple (x_i * mask[:, 0] for x_i in torch.unbind (fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor (tp, axes, keepdim=False)
    fp = sum_tensor (fp, axes, keepdim=False)
    fn = sum_tensor (fn, axes, keepdim=False)

    return tp, fp, fn


class BDLoss (nn.Module):
    def __init__(self):
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super (BDLoss, self).__init__ ()
        # self.do_bg = do_bg

    def forward(self, net_output, target, bound):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        net_output = softmax_helper (net_output)
        # print('net_output shape: ', net_output.shape)
        pc = net_output[:, 1:, ...].type (torch.float32)
        dc = bound[:, 1:, ...].type (torch.float32)

        multipled = torch.einsum ("bcxyz,bcxyz->bcxyz", pc, dc)
        bd_loss = multipled.mean ()

        return bd_loss


class SoftDiceLoss (nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """

        """
        super (SoftDiceLoss, self).__init__ ()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list (range (2, len (shp_x)))
        else:
            axes = list (range (2, len (shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin (x)

        tp, fp, fn = get_tp_fp_fn (x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean ()

        return -dc


class DC_and_BD_loss (nn.Module):
    def __init__(self, soft_dice_kwargs, bd_kwargs, aggregate="sum"):
        super (DC_and_BD_loss, self).__init__ ()
        self.aggregate = aggregate
        self.bd = BDLoss (**bd_kwargs)
        self.dc = SoftDiceLoss (apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target, bound):
        dc_loss = self.dc (net_output, target)
        bd_loss = self.bd (net_output, target, bound)
        if self.aggregate == "sum":
            result = dc_loss + bd_loss
        else:
            raise NotImplementedError ("nah son")  # reserved for other stuff (later)
        return result


def compute_edts_forhdloss(segmentation):
    res = np.zeros (segmentation.shape)
    for i in range (segmentation.shape[0]):
        posmask = segmentation[i]
        negmask = ~posmask
        res[i] = distance_transform_edt (posmask) + distance_transform_edt (negmask)
    return res


def compute_edts_forPenalizedLoss(GT):
    """
    GT.shape = (batch_size, x,y,z)
    only for binary segmentation
    """
    res = np.zeros (GT.shape)
    for i in range (GT.shape[0]):
        posmask = GT[i]
        negmask = ~posmask
        pos_edt = distance_transform_edt (posmask)
        pos_edt = (np.max (pos_edt) - pos_edt) * posmask
        neg_edt = distance_transform_edt (negmask)
        neg_edt = (np.max (neg_edt) - neg_edt) * negmask

        res[i] = pos_edt / np.max (pos_edt) + neg_edt / np.max (neg_edt)
    return res


class DistBinaryDiceLoss (nn.Module):
    """
    Distance map penalized Dice loss
    Motivated by: https://openreview.net/forum?id=B1eIcvS45V
    Distance Map Loss Penalty Term for Semantic Segmentation
    """

    def __init__(self, smooth=1e-5):
        super (DistBinaryDiceLoss, self).__init__ ()
        self.smooth = smooth

    def forward(self, net_output, gt):
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        net_output = softmax_helper (net_output)
        # one hot code for gt
        with torch.no_grad ():
            if len (net_output.shape) != len (gt.shape):
                gt = gt.view ((gt.shape[0], 1, *gt.shape[1:]))

            if all ([i == j for i, j in zip (net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long ()
                y_onehot = torch.zeros (net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda (net_output.device.index)
                y_onehot.scatter_ (1, gt, 1)

        gt_temp = gt[:, 0, ...].type (torch.float32)
        with torch.no_grad ():
            dist = compute_edts_forPenalizedLoss (gt_temp.cpu ().numpy () > 0.5) + 1.0
        # print('dist.shape: ', dist.shape)
        dist = torch.from_numpy (dist)

        if dist.device != net_output.device:
            dist = dist.to (net_output.device).type (torch.float32)

        tp = net_output * y_onehot
        tp = torch.sum (tp[:, 1, ...] * dist, (1, 2, 3))

        dc = (2 * tp + self.smooth) / (torch.sum (net_output[:, 1, ...], (1, 2, 3)) + torch.sum (y_onehot[:, 1, ...], (
        1, 2, 3)) + self.smooth)

        dc = dc.mean ()

        return -dc

