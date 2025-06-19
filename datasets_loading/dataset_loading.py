import os
from torch.utils.data import Dataset as dataset_torch
import numpy as np
import SimpleITK as sitk
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.transform import resize
import cv2
import random
import os


mean = np.array([0.5,])
std = np.array([0.5,])


def augment_data(image, angle, tx, ty):
    # Perform rotation
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))

    # Perform translation
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(rotated_image, M, (cols, rows))

    return translated_image

def inverse_transform(prediction, angle, tx, ty, flags=True):
    rows, cols = prediction.shape[:2]
    if flags == True:
        # Inverse translation
        M = np.float32([[1, 0, -tx], [0, 1, -ty]])
        translated_prediction = cv2.warpAffine(prediction, M, (cols, rows), flags=cv2.INTER_LINEAR)

        # Inverse rotation
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -float(angle), 1)
        original_prediction = cv2.warpAffine(translated_prediction, M, (cols, rows), flags=cv2.INTER_LINEAR)
    else:
        # Inverse translation
        M = np.float32([[1, 0, -tx], [0, 1, -ty]])
        translated_prediction = cv2.warpAffine(prediction, M, (cols, rows), flags=cv2.INTER_NEAREST)

        # Inverse rotation
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -float(angle), 1)
        original_prediction = cv2.warpAffine(translated_prediction, M, (cols, rows), flags=cv2.INTER_NEAREST)

    return original_prediction


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(1):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


import os

from sklearn.model_selection import KFold
def _make_image_namelist(dir, mode, fold_index=0):
    # Step 1: Collect all patient names and their MRI paths
    patient_dict = {}

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            splitted = fname.split('_')
            if 'T1' in splitted and 'label' not in splitted:
                patient_name = '_'.join(splitted[:-2])
                mri_path = os.path.join(root, fname)

                if os.path.exists(mri_path.replace('T1', 'DE')):
                    if patient_name not in patient_dict:
                        patient_dict[patient_name] = []
                    patient_dict[patient_name].append(mri_path.replace('T1', 'DE'))

    # Step 2: Split patients into 4 folds
    patients = list(patient_dict.keys())

    print(patients)

    kf = KFold(n_splits=5, shuffle=True, random_state=24)
    splits = list(kf.split(patients))

    if fold_index < 0 or fold_index >= 5:
        raise ValueError("fold_index must be between 0 and 3")

    train_patients = set()
    test_patients = set()

    for i, (train_index, test_index) in enumerate(splits):
        if i == fold_index:
            test_patients.update([patients[idx] for idx in test_index])
        else:
            train_patients.update([patients[idx] for idx in train_index])

    train_patients = list(set(train_patients) - set(test_patients))

    # Step 3: Collect paths for the chosen patients
    def collect_paths(patient_list):
        MRI_path = []
        namelist = []
        for patient in patient_list:
            for path in patient_dict[patient]:
                MRI_path.append(path)
                namelist.append(os.path.basename(path))
        return MRI_path, namelist

    print("training cases", len(train_patients), list(train_patients))
    print("val cases", len(test_patients), list(test_patients))

    if mode == 'train':
        return collect_paths(train_patients)
    else:
        return collect_paths(test_patients)


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
    def __init__(self, root, mode='train', fold_index=0):
        self.root = root
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.img_paths, self.img_names = _make_image_namelist(os.path.join(self.root), self.mode, fold_index)

        self.epi = 0
        self.img_num = len(self.img_names)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )

        print(self.mode, self.img_num)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):

        path_pdff = self.img_paths[index]
        case_name = self.img_names[index]
        name = case_name[0:-4]

        img_mri_t1 = sitk.ReadImage(path_pdff.replace('DE', 'T1'))
        img_mri_t2 = sitk.ReadImage(path_pdff.replace('DE', 'T1'))
        img_pdff = sitk.ReadImage(path_pdff)
        label_mri = sitk.ReadImage(path_pdff.replace('DE', 'T1_label'))
        img_US = sitk.ReadImage(path_pdff.replace('DE', 'US'))

        PDFF_image_mat = np.array(sitk.GetArrayFromImage(img_pdff)).astype(float)
        t1_image_mat = np.array(sitk.GetArrayFromImage(img_mri_t1)).astype(float)
        t2_image_mat = np.array(sitk.GetArrayFromImage(img_mri_t2)).astype(float)
        MR_label_mat = np.array(sitk.GetArrayFromImage(label_mri)).astype(float)
        US_image_mat = np.array(sitk.GetArrayFromImage(img_US)).astype(float)

        liver_labels = np.array(MR_label_mat == 3.0).astype(float)
        US_masks = np.array(US_image_mat > 0).astype(float)

        t1_image_mat = (t1_image_mat - t1_image_mat.min()) / (
                    t1_image_mat.max() - t1_image_mat.min() + 0.000001)

        t2_image_mat = (t2_image_mat - t2_image_mat.min()) / (
                t2_image_mat.max() - t2_image_mat.min() + 0.000001)

        US_image_mat = US_image_mat / 255
        PDFF_image_mat = PDFF_image_mat * 2


        cood = np.floor(US_image_mat.shape[0]/2)

        if self.mode == 'train':
            slice_index = np.random.randint(cood-30, cood+30)
        else:
            slice_index = np.random.randint(cood - 1, cood + 1)

        MR_t1 = t1_image_mat[slice_index, :, :]
        MR_t2 = t2_image_mat[slice_index, :, :]
        PDFF = PDFF_image_mat[slice_index, :, :]
        US = US_image_mat[slice_index, :, :]
        liver_label = liver_labels[slice_index, :, :]
        US_mask = US_masks[slice_index, :, :]

        MR_t1 = crop_and_pad(MR_t1)
        MR_t2= crop_and_pad(MR_t2)
        PDFF = crop_and_pad(PDFF)
        US = crop_and_pad(US)
        liver_label = crop_and_pad(liver_label)
        US_mask = crop_and_pad(US_mask)

        MR_t1 = resize(MR_t1, (256, 256), anti_aliasing=True)
        MR_t2 = resize(MR_t2, (256, 256), anti_aliasing=True)
        PDFF = resize(PDFF, (256, 256), anti_aliasing=True)
        US = resize(US, (256, 256), anti_aliasing=True)
        liver_label = resize(liver_label, (256, 256), order=0, anti_aliasing=False)
        US_mask = resize(US_mask, (256, 256), order=0, anti_aliasing=False)

        US_augs = []
        angles = np.zeros([10, 1])
        translations = np.zeros([10, 2])

        for i in range(0, 5):
            angle = random.uniform(-10, 10)
            translation = (random.uniform(-5, 5), random.uniform(-5, 5))
            US_aug = augment_data(US, angle, translation[0], translation[1])
            US_augs.append(self.transform(US_aug))
            angles[i, 0] = angle
            translations[i, 0] = translation[0]
            translations[i, 1] = translation[1]
        US_augs = torch.cat(US_augs, 0)

        MR_t1 = self.transform(MR_t1 * US_mask)
        MR_t2 = self.transform(MR_t2 * US_mask)
        PDFF = self.transform(PDFF * US_mask)
        US = self.transform(US * US_mask)

        liver_label = np.expand_dims(liver_label * US_mask, 0)

        imgs = {'T1': MR_t1, 'T2':MR_t2, 'PDFF': PDFF, 'US': US, 'US_augs': US_augs, 'angles': angles, 'trans': translations, 'label': liver_label, 'name': name + '_' + str(slice_index)}
        return imgs
