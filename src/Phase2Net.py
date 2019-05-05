import torch
import torch.nn as nn
from torchvision import models

def get_model():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    model = nn.Sequential(model_ft, nn.Sigmoid())
    return model_ft