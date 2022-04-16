# Original Author       : Ghasem Abdi, ghasem.abdi@yahoo.com
# File Last Update Date : April 15, 2022

import os
import glob
import cv2 as cv
import numpy as np
import albumentations as A

from typing import Optional
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2

class prepare_data(Dataset):
    """prepare a custom dataset for change detection according the following data structure:
    ├── data_root
    │   ├── train
    │   │   ├── base_dir
    │   │   ├── target_dir
    │   │   ├── label_dir
    │   ├── val
    │   │   ├── base_dir
    │   │   ├── target_dir
    │   │   ├── label_dir
    │   ├── optional [test]
    │   │   ├── base_dir
    │   │   ├── target_dir
    │   │   ├── optional [label_dir]
    data_root: a string path of change detection data structure,
    base_dir: a string name of the base images directory, default is 'A',
    base_img_suffix: a string indicator of the base images suffix, default is '*.png',
    target_dir: a string name of the target images directory, default is 'B',
    target_img_suffix: a string indicator of the target images suffix, default is '*.png',
    label_dir: a string name of the ground truth masks ('no change: 0', 'change: 255') directory, default is 'label',
    label_mask_suffix: a string indicator of the ground truth masks suffix, default is '*.png',
    size: an integer of the model input size, default is 256,
    transform: a combination of albumentations library transforms for data augmentation, default is None"""

    def __init__(
        self, 
        data_root: str, 
        base_dir: str = 'A', 
        base_img_suffix: str = '*.png', 
        target_dir: str = 'B', 
        target_img_suffix: str = '*.png', 
        label_dir: Optional[str] = 'label', 
        label_mask_suffix: Optional[str] = '*.png', 
        size: int = 256,  
        transform = None
    ):
        super(prepare_data, self).__init__()
        
        self.base_img = glob.glob(pathname=os.path.join(data_root, base_dir, base_img_suffix))
        self.target_img = glob.glob(pathname=os.path.join(data_root, target_dir, target_img_suffix))
        self.label_mask = glob.glob(pathname=os.path.join(data_root, label_dir, label_mask_suffix)) \
            if label_dir else None

        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.base_img)

    def __getitem__(self, idx):
        fname = self.base_img[idx]
        base = cv.cvtColor(src=cv.imread(filename=self.base_img[idx]), code=cv.COLOR_BGR2RGB)
        target = cv.cvtColor(src=cv.imread(filename=self.target_img[idx]), code=cv.COLOR_BGR2RGB)
        if self.label_mask: mask = cv.imread(filename=self.label_mask[idx], flags=cv.IMREAD_GRAYSCALE) / 255

        additional_targets = {'image_2':'image', 'label':'mask'} if self.label_mask else {'image_2':'image'}

        if self.transform is None: #default transformation for train set
            self.transform = A.Compose(transforms=[A.Resize(height=self.size, width=self.size), A.Flip(), A.Rotate(limit=15), \
                A.Normalize(), ToTensorV2()], additional_targets=additional_targets)

        elif self.transform is False: #default transformation for valid/test set
            self.transform = A.Compose(transforms=[A.Resize(height=self.size, width=self.size), A.Normalize(), ToTensorV2()], \
                additional_targets=additional_targets)

        if self.label_mask:
            sample = self.transform(image=base, image_2=target, label=mask)
            return sample['image'], sample['image_2'], sample['label'], fname
        
        else:
            sample = self.transform(image=base, image_2=target)
            return sample['image'], sample['image_2'], fname


    def show_samples(self, n=5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), figsize=(15, 25), **kwargs):
        _, ax = plt.subplots(nrows=n, ncols=3, figsize=figsize) if self.label_mask \
            else plt.subplots(nrows=n, ncols=2, figsize=figsize)
        
        for i in range(n):
            if self.label_mask:
                base, target, mask, _ = self.__getitem__(np.random.randint(low=0, high=(self.__len__())))
            
            else:
                base, target, _ = self.__getitem__(np.random.randint(low=0, high=(self.__len__())))

            ax[i, 0].set_axis_off(), ax[i, 0].imshow(X=(std * base.permute(1, 2, 0).numpy() + mean).clip(min=0, max=1), **kwargs)
            ax[i, 1].set_axis_off(), ax[i, 1].imshow(X=(std * target.permute(1, 2, 0).numpy() + mean).clip(min=0, max=1), **kwargs)
            if self.label_mask: ax[i, 2].set_axis_off(), ax[i, 2].imshow(X=mask, **kwargs)