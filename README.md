# SSTB: Self-Supervised T2WI-Bridged Framework for PDFF Prediction

This repository hosts documentation and controlled-access request information for our SSTB framework.

Due to ongoing IP agreements, pretrained models are available upon request for **non-commercial academic research** only.

🔗 **Request form**: [[Google request form](https://docs.google.com/forms/d/e/1FAIpQLSds-WYsSX7jaM6EVQFzyb7AwtfYgk3hCcZBr3_IivItah9wYQ/viewform?usp=sharing&ouid=100817192501032993715)]

📃 Access is granted under a Non-Commercial Research License. See `Access_Policy.md` for details.

The training includes two steps:
**1. Self-supervised learning on US images"
The 'main_pretext.py" uses grayscale US images for training and validation. Please follow these instructions to prepare your data:

i. Data Organization
Place all your ultrasound images in a single folder, e.g., data_root/.
All images must be in .jpg format and should be grayscale.

Example directory structure:
data_root/
    img001.jpg
    img002.jpg
    img003.jpg
    ...
ii. File Naming
Image filenames can be arbitrary, but must end with .jpg.


**2. Supervised learning on US-PDFF images"
The "main.py" requires multimodal medical images for each subject, including MRI T2, PDFF, ultrasound (US), and label images. Please organize your data as follows:

1. Directory Structure
Organize all subject data under a root directory (e.g., data_root/). There should be no subfolders per subject; all files are placed directly under the root.

Each subject must have files for:

T2 MRI
PDFF
Ultrasound (US)
Label (liver segmentation)

2. File Naming Convention
Each file should be named using the pattern:

[subject_id]_T2.nii.gz         # T2-weighted MRI image
[subject_id]_DE.nii.gz         # PDFF image (DE = Dixon-Encoded or similar)
[subject_id]_US.nii.gz         # Ultrasound image
[subject_id]_T1_label.nii.gz   # Segmentation label for T1 image

Example:

data_root/
    patient001_T1.nii.gz
    patient001_T2.nii.gz
    patient001_DE.nii.gz
    patient001_US.nii.gz
    patient001_T1_label.nii.gz
    patient002_T1.nii.gz
    patient002_T2.nii.gz
    patient002_DE.nii.gz
    patient002_US.nii.gz
    patient002_T1_label.nii.gz
    ...
3. Image Format
All files must be in NIfTI format (.nii.gz), compatible with SimpleITK.

Ensure all images for each subject have the same spatial dimensions and alignment.

4. Folds and Data Splitting
The dataset will be automatically split into 5 folds using the patient IDs.

By default, 4 folds are used for training and 1 fold for validation/testing.

Set the fold_index parameter (0-4) in the code to select the validation fold.

