import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import numbers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GaussianSmoothing_withPad(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2, device=None):
        super(GaussianSmoothing_withPad, self).__init__()
        self.ksz = kernel_size
        self.device = device
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        pad_size = self.ksz // 2
        if len(input.shape) == 4:  # 2D
            input = F.pad(input, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        elif len(input.shape) == 5:  # 3D
            input = F.pad(input, (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size), mode='reflect')
        return self.conv(input, weight=self.weight.to(self.device), groups=self.groups)



class MSE_blur_loss(nn.Module):
    def __init__(self, channels=1, kernel_size=5, sigma=1, dim=2):
        super(MSE_blur_loss, self).__init__()
        print('MSE_blur Loss')
        self.Gaussian_blur = GaussianSmoothing_withPad(channels, kernel_size, sigma, dim)
        self.kernel_size = kernel_size
        self.cri_pix = nn.MSELoss()

    def forward(self, x, y):
        x = self.Gaussian_blur(x)
        y = self.Gaussian_blur(y)
        loss = self.cri_pix(x, y)
        return loss


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


class bce_dice_loss(nn.Module):
    def __init__(self):
        super(bce_dice_loss, self).__init__()
        self.criterion = nn.BCELoss(weight=None, size_average=True, reduce=True)

    def forward(self, y_pred, y_true):
        smooth = 1.
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        # dice_loss

        intersection = (y_true * y_pred).sum()
        dice_loss = (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)
        dice_loss = 1.0 - dice_loss
        # binary_loss
        binary_loss = self.criterion(y_pred, y_true)

        return binary_loss + dice_loss


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred = pred.squeeze(dim=1)

        smooth = 1

        dice = 0.

        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1) / (
                        pred[:, i].pow(2).sum(dim=1).sum(dim=1) +
                        target[:, i].pow(2).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 1)


class ELDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        dice = 0.

        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1) / (
                        pred[:, i].pow(2).sum(dim=1).sum(dim=1) +
                        target[:, i].pow(2).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)

        return torch.clamp((torch.pow(-torch.log(dice + 1e-5), 0.3)).mean(), 0, 2)


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce_loss = nn.BCELoss()
        self.bce_weight = 1.0

    def forward(self, pred, target):
        smooth = 1

        dice = 0.

        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1) / (
                        pred[:, i].pow(2).sum(dim=1).sum(dim=1) +
                        target[:, i].pow(2).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 1) + self.bce_loss(pred, target) * self.bce_weight


class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1


        jaccard = 0.

        for i in range(pred.size(1)):
            jaccard += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1) / (
                        pred[:, i].pow(2).sum(dim=1).sum(dim=1) +
                        target[:, i].pow(2).sum(dim=1).sum(dim=1) - (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1) + smooth)

        jaccard = jaccard / pred.size(1)
        return torch.clamp((1 - jaccard).mean(), 0, 1)


class SSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        loss = 0.

        for i in range(pred.size(1)):
            s1 = ((pred[:, i] - target[:, i]).pow(2) * target[:, i]).sum(dim=1).sum(dim=1) / (
                        smooth + target[:, i].sum(dim=1).sum(dim=1))

            s2 = ((pred[:, i] - target[:, i]).pow(2) * (1 - target[:, i])).sum(dim=1).sum(dim=1) / (
                        smooth + (1 - target[:, i]).sum(dim=1).sum(dim=1))

            loss += (0.05 * s1 + 0.95 * s2)

        return loss / pred.size(1)


class TverskyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1

        dice = 0.

        for i in range(pred.size(1)):
            dice += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1) / (
                        (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1) +
                        0.3 * (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1) + 0.7 * (
                                    (1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1) + smooth)

        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 2)


class L1_loss(nn.Module):
    def __init__(self, ):
        super(L1_loss, self).__init__()
        self.loss_f = nn.L1Loss().to(device)

    def forward(self, prediction, target):
        target = F.interpolate(target, size=prediction.size()[2:], mode='bilinear', align_corners=True)
        return self.loss_f(prediction, target)


class structure_loss(torch.nn.Module):
    def __init__(self):
        super(structure_loss, self).__init__()

    def _structure_loss(self, pred, mask):

        mask = F.interpolate(mask, size=pred.size()[2:], mode='bilinear', align_corners=True)
        weit = 1

        # wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        # wbce = (weit * wbce).sum(dim=(2, 3, 4)) / weit.sum(dim=(2, 3, 4))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter) / (union - inter)
        return wiou

    def forward(self, pred, mask):
        return self._structure_loss(pred, mask)


class MultiFrameDenoisingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, x_noisy):
        # 计算 MSE
        mse_loss = 0
        for i in range(y_pred.shape[1]):
            for j in range(y_pred.shape[1]):
                if i != j:
                    mse_loss += (y_pred[:, i, :, :, :] - x_noisy[:, j, :, :, :]).pow(2).mean()

        mse_loss /= (y_pred.shape[1] * (y_pred.shape[1] - 1))
        msa_loss = 0
        for i in range(y_pred.shape[1]):
            for j in range(i + 1, y_pred.shape[1]):
                msa_loss += (y_pred[:, i, :, :, :] - y_pred[:, j, :, :, :]).pow(2).mean()

        msa_loss /= (y_pred.shape[1] * (y_pred.shape[1] - 1) / 2)

        return mse_loss + msa_loss
