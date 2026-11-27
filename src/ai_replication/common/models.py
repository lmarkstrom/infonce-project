"""Encoder architectures specified in Section 5 and Appendix E.1.

Architectures:
1. LinearEncoder: single linear layer for synthetic experiments (Laplace, GMM, Binary).
2. TwoLayerMLPEncoder: two-layer MLP with a single ReLU activation for CIFAR-10 dynamics (Fig. 4).
3. ResNet18SimCLR & ResNet18Supervised: ResNet-18 encoders sharing identical backbone
   initialization for contrastive vs. supervised comparison on CIFAR-10 (Table 1).
"""

from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class LinearEncoder(nn.Module):
    """Simple linear encoder f: R^{d_in} -> R^{d_out} for synthetic experiments."""

    def __init__(self, input_dim: int = 1024, output_dim: int = 256, bias: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TwoLayerMLPEncoder(nn.Module):
    """Two-layer MLP with a single ReLU nonlinear activation for CIFAR-10 (Fig. 4).

    Maps flattened 3x32x32 CIFAR-10 images (3072 dims) through:
        Linear(3072, hidden_dim) -> ReLU -> Linear(hidden_dim, repr_dim)
    """

    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        hidden_dim: int = 512,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_resnet18_backbone(
    pretrained: bool = True,
    cifar_stem: bool = False,
) -> Tuple[nn.Module, int]:
    """Construct a ResNet-18 backbone outputting 512-dim representations.

    Appendix E.1 specifies a ResNet-18 encoder pretrained on ImageNet for CIFAR-10
    contrastive vs. supervised comparison. When `pretrained=False` and `cifar_stem=True`,
    uses the classic CIFAR 3x3 stride-1 stem without maxpool.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = models.resnet18(weights=weights)
    feature_dim = backbone.fc.in_features
    if cifar_stem:
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    return backbone, int(feature_dim)


class ResNet18SimCLR(nn.Module):
    """ResNet-18 encoder following the SimCLR protocol (512-dim backbone + 2-layer MLP projector).

    Contrastive loss operates on projected embeddings `z = projector(h)`, while
    representation diagnostics are evaluated on the 512-dim backbone output `h`
    (matching the 512-feature compliance ratios in Table 1).
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 512,
        proj_hidden_dim: int = 512,
        proj_output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        z = self.projector(h)
        return h, z


class ResNet18Supervised(nn.Module):
    """ResNet-18 supervised classifier sharing the exact backbone architecture as ResNet18SimCLR."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 512,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.classifier(h)
        return h, logits


def build_matched_resnet18_pair(
    pretrained: bool = True,
    cifar_stem: bool = False,
    num_classes: int = 10,
    proj_hidden_dim: int = 512,
    proj_output_dim: int = 128,
) -> Tuple[ResNet18SimCLR, ResNet18Supervised]:
    """Instantiate contrastive and supervised ResNet-18 models with IDENTICAL initial weights.

    Section 5 (Page 10): 'Both models share identical initialization and capacity,
    differing only in the objective: cross-entropy supervision versus InfoNCE.'
    """
    backbone_contrastive, feat_dim = build_resnet18_backbone(
        pretrained=pretrained,
        cifar_stem=cifar_stem,
    )
    backbone_supervised = copy.deepcopy(backbone_contrastive)
    contrastive_model = ResNet18SimCLR(
        backbone=backbone_contrastive,
        feature_dim=feat_dim,
        proj_hidden_dim=proj_hidden_dim,
        proj_output_dim=proj_output_dim,
    )
    supervised_model = ResNet18Supervised(
        backbone=backbone_supervised,
        feature_dim=feat_dim,
        num_classes=num_classes,
    )
    return contrastive_model, supervised_model

