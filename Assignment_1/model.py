### MODEL 3: ResNet 50
import torch.nn as nn
from torchvision import models

classes = ['bug', 'electric', 'fighting', 'fire', 'flying', 'grass', 'ground', 'phychic', 'poison', 'water']

class Model(nn.Module):
    def __init__(self, num_classes=len(classes), pretrained=True):
        super(Model, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        # freeze the first layer
        for param in self.model.layer1.parameters():
          param.requires_grad = False
        # add fully-connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(0.3),
                                      nn.Linear(num_ftrs, num_classes))

    def forward(self, x):
        return self.model(x)