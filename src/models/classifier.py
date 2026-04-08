import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b2,
    EfficientNet_B2_Weights,
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)


class _ViTBackbone(nn.Module):
    """Runs ViT up to (but not including) the classification head, returns CLS token."""

    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        # _process_input: conv_proj + reshape into token sequence + prepend CLS + add pos embed
        x = self.vit._process_input(x)
        cls = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.vit.encoder(x)
        return x[:, 0]  # CLS token


class RisClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.3,
        arch: str = "efficient_net_b2",
        num_extra_features: int = 0,
    ):
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.num_extra_features = num_extra_features

        if arch == "efficient_net_b2":
            base = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
            in_features = base.classifier[1].in_features
            # Everything up to (but not including) the final linear
            self.backbone = nn.Sequential(base.features, base.avgpool, nn.Flatten())
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features + num_extra_features, num_classes),
            )

        elif arch == "vit_b_16":
            base = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            in_features = base.heads.head.in_features
            self.backbone = _ViTBackbone(base)
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features + num_extra_features, num_classes),
            )

        elif arch == "convnext_tiny":
            base = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            in_features = base.classifier[2].in_features
            # classifier[0]=LayerNorm2d, classifier[1]=Flatten — both before the linear
            self.backbone = nn.Sequential(
                base.features, base.avgpool, base.classifier[0], base.classifier[1]
            )
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features + num_extra_features, num_classes),
            )

        else:
            raise ValueError(f"Unknown arch: {arch}")

    def forward(self, x, extra_features=None):
        features = self.backbone(x)
        if self.num_extra_features > 0 and extra_features is not None:
            features = torch.cat([features, extra_features], dim=1)
        return self.head(features)

    def freeze_backbone(self):
        """Phase 1: freeze everything, then unfreeze only the new head."""
        for param in self.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze(self):
        """Phase 2: fine-tune the entire network."""
        for param in self.parameters():
            param.requires_grad = True


def make_model(cfg, num_classes: int) -> RisClassifier:
    return RisClassifier(
        num_classes=num_classes,
        dropout=cfg["model"].get("dropout", 0.3),
        arch=cfg["model"].get("arch", "efficient_net_b2"),
        num_extra_features=cfg["model"].get("n_extra_features", 0),
    )
