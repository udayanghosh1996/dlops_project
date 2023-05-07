from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset, random_split
import os

# DATA_ROOT_PATH = r"F:\MTech_IIT_Jodhpur\3rd_Sem\DL-Ops\Project\DLOps_Project\DataPrep\datasets"

# DATA_ROOT_PATH = os.path.join(os.path.join(os.getcwd(), 'DataPrep'), 'dataset')
DATA_ROOT_PATH = './DataPrep/dataset'
os.makedirs(DATA_ROOT_PATH, exist_ok=True)


class ClassiferData(Dataset):
    def __init__(self, dataset_name, task):
        valid_models = ["cifar10", "cifar100"]
        if dataset_name.lower() not in valid_models:
            raise ValueError(f"The data should be in {valid_models}")

        data_src = CIFAR10 if dataset_name.lower() == "cifar10" else CIFAR100

        c_train = data_src(DATA_ROOT_PATH, download=True, train=True)
        c_train, c_val = random_split(c_train, [0.8, 0.2])
        c_train = c_train.dataset
        c_val = c_val.dataset
        c_test = data_src(DATA_ROOT_PATH, download=True, train=False)
        if task.strip().lower() == "train":
            self.all_img_np = c_train.data
            self.labels = c_train.targets
        elif task.strip().lower() == "val":
            self.all_img_np = c_val.data
            self.labels = c_val.targets
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
