"""
Encoder architectures used in the paper:

1. LinearEncoder   — single linear layer (synthetic experiments)
2. MLP             — two-layer MLP with single ReLU (CIFAR-10 dynamics, Fig. 4)
3. ResNet18Encoder — SimCLR-style ResNet-18 with projection head (CIFAR-10, Table 1)
4. ResNet18Supervised — ResNet-18 for supervised cross-entropy baseline

All encoders output unnormalized representations of dimension `out_dim`.
L2 normalization for InfoNCE is applied inside the loss function.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class LinearEncoder(nn.Module):
    """
    Linear encoder: f(x) = Wx + b.

    Used for synthetic data experiments (Section 5, Synthetic data).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLP(nn.Module):
    """
    Two-layer MLP with a single ReLU nonlinearity.

    Used for CIFAR-10 training dynamics experiment (Figure 4):
      x  →  Linear(in_dim, hidden_dim)  →  ReLU  →  Linear(hidden_dim, out_dim)

    Following the paper: "a two-layer MLP with a single ReLU nonlinear activation".
    For CIFAR-10, in_dim = 3*32*32 = 3072.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dims if needed (CIFAR-10 images arrive as (B, 3, 32, 32))
        x = x.view(x.shape[0], -1)
        return self.net(x)


class ResNet18Encoder(nn.Module):
    """
    ResNet-18 encoder following the SimCLR protocol.

    The classification head is replaced by a 2-layer projection MLP
    (as in SimCLR: Chen et al., 2020):
      backbone_features  →  Linear(512, 512)  →  ReLU  →  Linear(512, out_dim)

    Used for CIFAR-10 contrastive vs. supervised comparison (Table 1).
    """

    def __init__(self, out_dim: int = 256, pretrained: bool = False):
        super().__init__()

        # Load ResNet-18 backbone
        backbone = tv_models.resnet18(weights=None)

        # Remove the final classification layer
        self.feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Projection head (SimCLR-style)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        z = self.projector(h)
        return z

    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone (pre-projection) features for evaluation."""
        return self.backbone(x)


class ResNet18Supervised(nn.Module):
    """
    ResNet-18 for supervised cross-entropy baseline on CIFAR-10.

    Standard ResNet-18 with a linear classification head (10 classes for CIFAR-10).
    Used for the supervised comparison in Table 1.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        backbone = tv_models.resnet18(weights=None)
        feature_dim = backbone.fc.in_features  # 512
        # Replace fc with a classifier
        backbone.fc = nn.Linear(feature_dim, num_classes)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-classification representations for Gaussian diagnostics."""
        # Pass through all layers except the final fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
