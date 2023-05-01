import torch
import torch.nn as nn
import math


class NTXent(nn.Module):
    def __init__(self, batch_size:int, temp):
        super(NTXent, self).__init__()
        self.batch_size = batch_size
        self.temp = temp
        self.criterion = nn.CrossEntropyLoss()

    def mask_diagonal(self):
        n = 2 * self.batch_size
        mask_d = torch.ones((n, n), dtype=bool)
        for i in range(n):
            mask_d[i, i] = 0
        return mask_d

    def mask_pos(self):
        n = 2 * self.batch_size
        org_st = 0
        aug_st = self.batch_size
        mask_aug = torch.zeros((n, n), dtype=bool)
        i = 0
        while i < n/2:
            mask_aug[org_st, aug_st] = 1
            mask_aug[aug_st, org_st] = 1
            org_st += 1
            aug_st += 1
            i += 1
        return mask_aug

    def forward(self, z_org, z_aug):
        # print(z_org.shape)
        # print(z_aug.shape)
        # print(self.batch_size)
        n = 2 * self.batch_size
        mask_d = self.mask_diagonal()
        mask_pos = self.mask_pos()
        mask_neg = torch.ones((n, n), dtype=bool)
        mask_neg = torch.logical_and(mask_neg, torch.logical_not(mask_d))
        mask_neg = torch.logical_not(torch.logical_or(mask_neg, mask_pos))
        z = torch.cat((z_org, z_aug), dim=0)
        # print(z.shape)
        sim_mat = torch.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_mat /= self.temp
        # print(sim_mat.shape)
        negative_samples = sim_mat[mask_neg].reshape(n, -1)
        positive_samples = sim_mat[mask_pos].reshape(n, -1)
        logits = torch.cat((negative_samples, positive_samples), dim=1)
        # print(logits.shape)
        labels = torch.full((n,), n-2).to(positive_samples.device)
        loss = self.criterion(logits, labels)
        return loss

