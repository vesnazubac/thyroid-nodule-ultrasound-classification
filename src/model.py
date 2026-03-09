import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm  
import torch

class BaselineCNN(nn.Module):
    """
    Osnovni CNN sa 3 konvoluciona sloja.
    Ulaz: slike 3x128x128
    """
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DenseNet121Transfer(nn.Module):
    """
    DenseNet121 prethodno treniran na ImageNet-u.
    Ulaz: slike 3x224x224
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.densenet121(pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class ResNet50Transfer(nn.Module):
    """
    ResNet50 prethodno treniran na ImageNet-u.
    Ulaz: 3x224x224
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class EfficientNetB0Transfer(nn.Module):
    """
    EfficientNet-B0 (timm) prethodno treniran na ImageNet-u.
    Ulaz: 3x224x224
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class CheXNetDenseNet121(nn.Module):
    """
    DenseNet121 treniran na medicinskim slikama (CheXNet).
    Preuzima se sa torch.hub.
    Ulaz: 3x224x224
    """
    def __init__(self, num_classes=2):
        super().__init__()
        # CheXNet je dostupan na hub-u
        self.backbone = torch.hub.load('nih-chest-xray/chexnet', 'chexnet', pretrained=True)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)