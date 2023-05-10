import numpy as np
import torch
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from SimCLR_Data import SimCLRDataset
from Classifier_data import ClassiferData
from SimCLRLoss import NTXent
from torch.optim import Adam
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from sklearn.metrics import top_k_accuracy_score
from torch.autograd import Variable

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
# SAVE_DIR = r"F:\MTech_IIT_Jodhpur\3rd_Sem\DL-Ops\Project\DLOps_Project\artifacts"

#SAVE_DIR = os.path.join("../model_weights")
SAVE_DIR = os.path.join(os.getcwd(), 'SimCLR')
os.makedirs(SAVE_DIR, exist_ok=True)


class ResNet18enc:
    def __init__(self, unfreez_layers=0):
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)

        self.model = self.model.to(DEVICE)
        self.__preprocess = weights.transforms()

        num_param_layers = len(list(self.model.parameters()))
        if unfreez_layers == -1:
            unfreez_layers = num_param_layers

        freez_layers = num_param_layers - unfreez_layers
        for p_idx, param in enumerate(self.model.parameters()):
            if p_idx < freez_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def __call__(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = self.__preprocess(x).to(DEVICE)
        # print(x.shape)
        # transforms.Resize(224)(x)
        x = x.to(DEVICE)
        x_op = self.model(x)
        return x_op


class ProjectionHead(torch.nn.Module):
    def __init__(self, proj_dim=128):
        super(ProjectionHead, self).__init__()
        self.activation = F.relu
        self.layer1 = torch.nn.Linear(1000, 500)
        self.layer2 = torch.nn.Linear(500, proj_dim)
        self.device = DEVICE

    def forward(self, x):
        x = x.to(self.device)
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x


class SimCLR:
    def __init__(self, unfreezed_enc_layers=5, proj_head_dim=128):
        self.base_enc = ResNet18enc(unfreezed_enc_layers)
        self.projection_head = ProjectionHead(proj_head_dim).to(DEVICE)

    def __call__(self, x):
        f = self.base_enc(x)
        g = self.projection_head(f)
        return g

    def save_model(self, save_path):
        enc_state_dict_path = os.path.join(save_path, "encoder_load_state")
        proj_state_dict_path = os.path.join(save_path, "projection_head_load_state")
        torch.save(self.base_enc.model.state_dict(), enc_state_dict_path)
        torch.save(self.projection_head.state_dict(), proj_state_dict_path)

    def load_model(self, save_path):
        enc_state_dict_path = os.path.join(save_path, "encoder_load_state")
        proj_state_dict_path = os.path.join(save_path, "projection_head_load_state")
        self.base_enc.model.load_state_dict(torch.load(enc_state_dict_path, map_location=DEVICE))
        self.projection_head.load_state_dict(torch.load(proj_state_dict_path, map_location=DEVICE))


class Classifier(torch.nn.Module):
    def __init__(self, n_classes, unfreezed_enc_layers=0, enc_dim=128):
        super(Classifier, self).__init__()
        self.feature_extractor = SimCLR(unfreezed_enc_layers, enc_dim)
        self.n_classes = n_classes
        self.clf_layer1 = torch.nn.Linear(1000, enc_dim).to(DEVICE)
        self.clf_layer2 = torch.nn.Linear(enc_dim, n_classes).to(DEVICE)
        self.activation1 = F.relu

    def forward(self, x):
        x = x.to(DEVICE)
        x = self.feature_extractor.base_enc(x)

        # layer1 of classifier
        x = self.clf_layer1(x)
        x = self.activation1(x)

        # layer2 of classifier
        x = self.clf_layer2(x)
        return x

    def pretext_train(self, dataset_name,
                      epochs, enc_lr, proj_lr, fine_tune_layers=-1, temperature=0.05,
                      batch_size=16):
        dataset = SimCLRDataset(dataset_name, batch_size)
        dataloader = DataLoader(dataset, batch_size=1,
                                num_workers=3,
                                pin_memory=True
                                )

        model = self.feature_extractor
        criterion = NTXent(batch_size, temperature)

        optim_list = [{"params": model.base_enc.model.parameters(), "lr": proj_lr}]
        if fine_tune_layers == -1:
            fine_tune_layers = len(list(model.base_enc.model.parameters()))
        for i in range(1, fine_tune_layers + 1):
            optim_list.append({"params": list(model.base_enc.model.parameters())[-i],
                               "lr": enc_lr})

        # print(optim_list)
        optim = Adam(
            optim_list,
            lr=proj_lr,
            weight_decay=1e-06
        )
        model.projection_head.train()
        model.base_enc.model.train()

        # gc.collect()
        # torch.cuda.empty_cache()
        for epoch in tqdm(range(epochs)):
            for batch_idx, (original_tensors, aug_tensors) in enumerate(dataloader, start=1):
                optim.zero_grad()
                original_tensors = original_tensors.squeeze(dim=0)
                aug_tensors = aug_tensors.squeeze(dim=0)
                original_Zs = model(original_tensors)
                aug_Zs = model(aug_tensors)
                loss = criterion(original_Zs, aug_Zs)
                loss.backward()
                optim.step()
                print(f"epoch {epoch + 1} batch - {batch_idx} loss = {loss.item()}")

            print(f"epoch {epoch} ---- {loss.item()}")
        model.save_model(SAVE_DIR)
        print("model saved")

    def load_pretexted_model(self):
        self.feature_extractor.load_model(SAVE_DIR)

    def save_model(self):
        self.feature_extractor.save_model(SAVE_DIR)
        layer1_state_dict_path = os.path.join(SAVE_DIR, "layer1")
        layer2_state_dict_path = os.path.join(SAVE_DIR, "layer2")
        torch.save(self.clf_layer1.state_dict(), layer1_state_dict_path)
        torch.save(self.clf_layer2.state_dict(), layer2_state_dict_path)

    def load_model(self):
        self.feature_extractor.load_model(SAVE_DIR)
        layer1_state_dict_path = os.path.join(SAVE_DIR, "layer1")
        layer2_state_dict_path = os.path.join(SAVE_DIR, "layer2")
        self.clf_layer1.load_state_dict(torch.load(layer1_state_dict_path, map_location=DEVICE))
        self.clf_layer2.load_state_dict(torch.load(layer2_state_dict_path, map_location=DEVICE))

    def fine_tuning(self, dataset_name, epochs, clf_lr,
                    batch_size=16):
        dataset = ClassiferData(dataset_name, "train")
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=3,
                                pin_memory=True,
                                shuffle=True
                                )

        val_dataset = ClassiferData(dataset_name, "val")
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=3,
            pin_memory=True,
            shuffle=True
            )
        num_of_batches = len(dataloader)
        cut_off_batch_cnt = int(num_of_batches * 0.01)

        model = self.feature_extractor
        criterion = CrossEntropyLoss()

        optim_list = []
        #optim_list = [{"params": model.projection_head.parameters(), "lr": proj_lr}]
        #for i in range(1, base_enc_finetune_layers + 1):
        #    optim_list.append({"params": list(model.base_enc.model.parameters())[-i],
        #                       "lr": enc_lr})
        optim_list.append({"params": list(self.clf_layer1.parameters()), "lr": clf_lr})
        optim_list.append({"params": list(self.clf_layer2.parameters()), "lr": clf_lr})

        optim = Adam(
            optim_list,
            lr=clf_lr,
            weight_decay=1e-06
        )

        model.projection_head.train()
        model.base_enc.model.train()

        for epoch in tqdm(range(epochs)):
            batch_no = 0
            batch_losses = []
            batch_accs = []
            batch_accs_10 = []
            self.clf_layer1.train()
            self.clf_layer2.train()
            for batch_data, batch_label in dataloader:
                if batch_no >= cut_off_batch_cnt:
                    break
                batch_data = batch_data.to(DEVICE)
                batch_label = batch_label.to(DEVICE)
                optim.zero_grad()
                z = model.base_enc(batch_data)
                z = self.clf_layer1(z)
                z = self.clf_layer2(z)
                loss = criterion(z, batch_label)
                loss.backward()
                optim.step()
                batch_no += 1
                print(f"epoch {epoch + 1} batch - {batch_no} loss = {loss.item()}")
                batch_losses.append(loss.item())

                b_acc = top_k_accuracy_score(
                    batch_label.cpu().numpy(),
                    z.detach().cpu().numpy(),
                    k=1,
                    labels=list(range(self.n_classes))
                )
                batch_accs.append(b_acc)

                b_acc_10 = top_k_accuracy_score(
                    batch_label.cpu().numpy(),
                    z.detach().cpu().numpy(),
                    k=10,
                    labels=list(range(self.n_classes))
                )
                batch_accs_10.append(b_acc_10)

            self.clf_layer1.eval()
            self.clf_layer2.eval()
            v_batch_accs = []
            v_batch_accs_10 = []
            for batch_data, batch_label in val_dataloader:
                batch_data = batch_data.to(DEVICE)
                batch_label = batch_label.to(DEVICE)
                z = model.base_enc(batch_data)
                z = self.clf_layer1(z)
                z = self.clf_layer2(z)
                batch_no += 1

                b_acc = top_k_accuracy_score(
                    batch_label.cpu().numpy(),
                    z.detach().cpu().numpy(),
                    k=1,
                    labels=list(range(self.n_classes))
                )
                v_batch_accs.append(b_acc)

                b_acc_10 = top_k_accuracy_score(
                    batch_label.cpu().numpy(),
                    z.detach().cpu().numpy(),
                    k=10,
                    labels=list(range(self.n_classes))
                )
                v_batch_accs_10.append(b_acc_10)

            print(f"epoch {epoch} ---- {np.mean(batch_losses)} \
            train_acc: {np.mean(batch_accs)} train_acc_top10: {np.mean(batch_accs_10)}\n Val_acc:\
            {np.mean(v_batch_accs)} Val_acc_top10: {np.mean(v_batch_accs_10)}\n")
        self.save_model()
        print("model saved")


if __name__ == "__main__":
    clf = Classifier(100)
    d = torch.rand(1, 3, 224, 224, dtype=torch.float32)
    res = clf(d)
    print(res.shape)
