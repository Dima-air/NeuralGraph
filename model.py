import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))

        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128)
        )

    def get_embedding(self, x):
        """
        Публичный метод для получения вектора признака одного изображения
        :param x:
        :return: output
        """
        output = self.cnn(x)
        output = self.adaptive_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        output = F.normalize(output, p=2, dim=1)
        return output

    def forward(self, input1, input2):
        output1 = self.get_embedding(input1)
        output2 = self.get_embedding(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_dis = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_dis, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_dis, min=0.0), 2))

        return loss_contrastive