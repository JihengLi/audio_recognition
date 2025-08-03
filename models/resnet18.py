import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet18Model(nn.Module):
    def __init__(self, embed_dim, hidden_size):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        self.projection = nn.Linear(hidden_size, embed_dim)

    def forward(self, input_features):
        features = self.resnet(input_features)
        emb = self.projection(features)
        emb = F.normalize(emb, p=2, dim=1)
        return emb
