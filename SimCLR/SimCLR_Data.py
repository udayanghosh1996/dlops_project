import os

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset


# DATA_ROOT_PATH = r"F:\MTech_IIT_Jodhpur\3rd_Sem\DL-Ops\Project\DLOps_Project\DataPrep\datasets"
DATA_ROOT_PATH = os.path.join(os.path.join(os.getcwd(), 'DataPrep'), 'dataset')


class SimCLRDataset(Dataset):
    def __init__(self, dataset_name, batch_size):
        self.batch_size = batch_size
        valid_models = ["cifar10", "cifar100"]
        if dataset_name.lower() not in valid_models:
            raise ValueError(f"The data should be in {valid_models}")

        data_src = CIFAR10 if dataset_name.lower() == "cifar10" else CIFAR100

        c_train = data_src(DATA_ROOT_PATH, download=True, train=True)
        c_test = data_src(DATA_ROOT_PATH, download=True, train=False)
        self.all_img_np = np.concatenate((c_train.data, c_test.data))

        s = 1.0
        size = self.all_img_np.shape[1]
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        gaussianblur = transforms.GaussianBlur(kernel_size=int(0.1 * size), sigma=(0.1, 2.0))

        self.data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=size),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomApply([gaussianblur], p=0.5)
        ])

        self.no_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.all_img_np) // self.batch_size

    def __getitem__(self, idx_):
        idxs = list(np.random.choice(len(self.all_img_np), self.batch_size, replace=False))
        original_tensors = []
        aug_tensors = []
        for idx in idxs:
            aug_tensors.append(self.data_transforms(self.all_img_np[idx]))
            original_tensors.append(self.no_transforms(self.all_img_np[idx]))
        return torch.stack(original_tensors), torch.stack(aug_tensors)


