from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset
import os

# DATA_ROOT_PATH = r"F:\MTech_IIT_Jodhpur\3rd_Sem\DL-Ops\Project\DLOps_Project\DataPrep\datasets"

DATA_ROOT_PATH = os.path.join(os.path.join(os.getcwd(), 'DataPrep'), 'dataset')


class ClassiferData(Dataset):
    def __init__(self, dataset_name, task):
        valid_models = ["cifar10", "cifar100"]
        if dataset_name.lower() not in valid_models:
            raise ValueError(f"The data should be in {valid_models}")

        data_src = CIFAR10 if dataset_name.lower() == "cifar10" else CIFAR100

        c_train = data_src(DATA_ROOT_PATH, download=True, train=True)
        c_test = data_src(DATA_ROOT_PATH, download=True, train=False)
        if task.strip().lower() == "train":
            self.all_img_np = c_train.data
            self.labels = c_train.targets
        else:
            self.all_img_np = c_test.data
            self.labels = c_test.targets

        self.no_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.all_img_np)

    def __getitem__(self, idx_):
        img = self.no_transforms(self.all_img_np[idx_])
        label = self.labels[idx_]
        return img, label
