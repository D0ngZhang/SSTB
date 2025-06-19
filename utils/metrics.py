import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff, cdist
import math
import cv2

class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)

class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets, n=1):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += n
        self.avg = np.around(self.sum / self.count, 4)

    @staticmethod
    def get_dices(logits, targets, threshold=0.5):
        """
        Computes the Dice coefficient for each class.

        Args:
            logits (torch.Tensor): Predicted logits of shape (batch_size, class_num, height, width).
            targets (torch.Tensor): Ground truth labels of shape (batch_size, class_num, height, width).
            threshold (float): Threshold to binarize the logits. Default is 0.5.

        Returns:
            np.ndarray: Dice coefficients for each class.
        """
        dices = []
        with torch.no_grad():
            # Apply sigmoid activation if logits are not probabilities
            if logits.max() > 1 or logits.min() < 0:
                logits = torch.sigmoid(logits)

            # Binarize logits
            logits = (logits > threshold).float()

            for class_index in range(targets.size(1)):
                inter = torch.sum(logits[:, class_index, :, :] * targets[:, class_index, :, :])
                union = torch.sum(logits[:, class_index, :, :]) + torch.sum(targets[:, class_index, :, :])
                dice = (2. * inter + 1) / (union + 1)
                dices.append(dice.item())
        return np.asarray(dices)


class AccuracyAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets, n=1):
        self.value = self.get_accuracies(logits, targets)
        self.sum += self.value
        self.count += n
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_accuracies(logits, targets):
        accuracies = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :] * targets[:, class_index, :, :])
            union = torch.sum(logits[:, class_index, :, :])
            accuracy = inter / (union + 1e-4)
            accuracies.append(accuracy.item())
        return np.asarray(accuracies)


class HausdorffDistance(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,):
        self.reset()

    def reset(self):
        self.value = np.asarray([0], dtype='float64')
        self.avg = np.asarray([0], dtype='float64')
        self.sum = np.asarray([0], dtype='float64')
        self.count = 0

    def update(self, logits, targets, n):
        self.value = self.get_accuracies(logits.cpu().data.numpy(), targets.cpu().data.numpy())
        self.sum += self.value
        self.count += n
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_accuracies(logits, targets):
        accuracies = []
        for i in range(logits.shape[0]):
            x = logits[i, 0, :, :]
            y = targets[i, 0, :, :]
            accuracy = directed_hausdorff(x, y)[0]
            accuracies.append(accuracy)
        return np.sum(accuracies)


class HD95(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,):
        self.reset()

    def reset(self):
        self.value = np.asarray([0], dtype='float64')
        self.avg = np.asarray([0], dtype='float64')
        self.sum = np.asarray([0], dtype='float64')
        self.count = 0

    def update(self, logits, targets, n):
        self.value = self.get_accuracies(logits.cpu().data.numpy(), targets.cpu().data.numpy())
        self.sum += self.value
        self.count += n
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_accuracies(logits, targets):
        batch_size = logits.shape[0]
        results = []

        for i in range(batch_size):
            # Get the foreground points (non-zero pixel coordinates)
            points1 = np.argwhere(logits[i, 0, :, :] > 0)  # Shape (n1, 2)
            points2 = np.argwhere(targets[i, 0, :, :] > 0)  # Shape (n2, 2)

            if points1.size == 0 or points2.size == 0:
                results.append(float('inf'))  # Handle empty masks by returning infinity
                continue

            # Compute pairwise distances between points
            dists_1_to_2 = cdist(points1, points2, 'euclidean')
            dists_2_to_1 = cdist(points2, points1, 'euclidean')

            # Find the 95th percentile of directed distances
            hd_95_1_to_2 = np.percentile(np.min(dists_1_to_2, axis=1), 95)
            hd_95_2_to_1 = np.percentile(np.min(dists_2_to_1, axis=1), 95)

            # The 95HD is the maximum of the two directed 95th percentile distances
            hd_95 = max(hd_95_1_to_2, hd_95_2_to_1)
            results.append(hd_95)

        return np.sum(results)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
