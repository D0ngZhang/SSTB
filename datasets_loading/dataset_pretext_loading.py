import os
import random

from torch.utils.data import Dataset as dataset_torch
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.transform import resize, rotate
import os


mean = np.array([0.5,])
std = np.array([0.5,])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(1):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


def augment_image(image):
    # Random horizontal flip
    if random.random() < 0.5:
        image = rotate(image, random.randint(1, 180))

    # # Random color jitter (brightness, contrast)
    # if random.random() < 0.8:
    #     alpha = random.uniform(0.8, 1.2)
    #     beta = random.uniform(-0.2, 0.2)
    #     image = np.clip(image * alpha + beta, 0, 1)
    #
    # # Random Gaussian blur
    # if random.random() < 0.5:
    #     mean = 0
    #     std = 0.1
    #     noise = np.random.normal(mean, std, image.shape)
    #     image = image + noise
    #     image = np.clip(image, 0, 1)

    return image


def random_crop(image, scale=(0.8, 1.0)):
    height, width = image.shape
    area = height * width
    for _ in range(10):
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect_ratio = random.uniform(3/4, 4/3)

        new_width = int(round(np.sqrt(target_area * aspect_ratio)))
        new_height = int(round(np.sqrt(target_area / aspect_ratio)))

        if new_width <= width and new_height <= height:
            x = random.randint(0, width - new_width)
            y = random.randint(0, height - new_height)
            return image[y:y+new_height, x:x+new_width]

    return resize(image, (int(width * scale[0]), int(height * scale[0])), anti_aliasing=True)


def _make_image_namelist(dir, mode):

    image_names = []

    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    image_files = [f for f in files if f.endswith('.jpg')]

    for image_file in image_files:
       image_names.append(image_file)

    total_num = len(image_names)
    training_num = int(total_num * 0.9)

    if mode == 'train':
        return image_names[0:training_num]
    else:
        return image_names[training_num:]

def crop_and_pad(array, target_size=(512, 512)):
    x, y = array.shape
    target_x, target_y = target_size

    center_x, center_y = x // 2, y // 2

    if x > target_x:
        start_x = center_x - target_x // 2
        end_x = start_x + target_x
    else:
        start_x = 0
        end_x = x

    if y > target_y:
        start_y = center_y - target_y // 2
        end_y = start_y + target_y
    else:
        start_y = 0
        end_y = y

    cropped_array = array[max(0, start_x):min(x, end_x), max(0, start_y):min(y, end_y)]

    if cropped_array.shape[0] < target_x or cropped_array.shape[1] < target_y:
        padding_x = (target_x - cropped_array.shape[0]) // 2
        padding_y = (target_y - cropped_array.shape[1]) // 2
        padded_array = np.pad(cropped_array,
                              ((padding_x, target_x - cropped_array.shape[0] - padding_x),
                               (padding_y, target_y - cropped_array.shape[1] - padding_y)),
                              mode='constant', constant_values=0)
    else:
        padded_array = cropped_array

    return padded_array

class data_set(dataset_torch):
    def __init__(self, root, mode='train'):
        self.root = root
        assert mode in ['train', 'val']
        self.mode = mode
        self.imgs_list = _make_image_namelist(os.path.join(self.root), self.mode)

        self.epi = 0
        self.img_num = len(self.imgs_list)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        print(self.mode, self.img_num)

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):

        img_name = self.imgs_list[index]
        index_neg = random.randint(0, len(self.imgs_list) - 1)

        while index_neg == index:
            index_neg = random.randint(0, len(self.imgs_list) - 1)

        img_name_neg = self.imgs_list[index_neg]

        name = img_name[:-4]

        path_p = os.path.join(self.root, img_name)
        path_n = os.path.join(self.root, img_name_neg)

        US_p = np.array(Image.open(path_p).convert('L'), dtype='float32')
        US_n = np.array(Image.open(path_n).convert('L'), dtype='float32')

        US_p = US_p / 255
        US_n = US_n / 255

        # MR = resize(MR, (192, 192), anti_aliasing=True)
        US_p = resize(US_p, (256, 256), anti_aliasing=True)
        US_n = resize(US_n, (256, 256), anti_aliasing=True)

        US_p2 = random_crop(US_p)
        US_p2 = resize(US_p2, (256, 256), anti_aliasing=True)
        US_p2 = augment_image(US_p2)

        US_n = random_crop(US_n)
        US_n = resize(US_n, (256, 256), anti_aliasing=True)
        US_n = augment_image(US_n)


        US_p = self.transform(US_p)
        US_p2 = self.transform(US_p2)
        US_n = self.transform(US_n)

        US_p = US_p.unsqueeze(0)
        US_p2 = US_p2.unsqueeze(0)
        US_n = US_n.unsqueeze(0)

        US = torch.cat((US_p, US_p2, US_n), 0)

        imgs = {"US": US, 'name': name}
        return imgs