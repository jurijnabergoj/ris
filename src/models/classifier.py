import torch.nn as nn
from torchvision.models import (
    efficientnet_b2,
    EfficientNet_B2_Weights,
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)


class RisClassifier(nn.Module):
    def __init__(
        self, num_classes: int, dropout: float = 0.3, arch: str = "efficientnet_b2"
    ):
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes

        if arch == "efficientnet_b2":
            self.model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        elif arch == "convnext_tiny":
            self.model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            # ConvNeXt classifier: [LayerNorm2d, Flatten, Linear]
            in_features = self.model.classifier[2].in_features
            self.model.classifier = nn.Sequential(
                self.model.classifier[0],  # LayerNorm2d
                self.model.classifier[1],  # Flatten
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            raise ValueError(f"Unknown arch: {arch}")

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        """Phase 1: only train the new head."""
        for param in self.model.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Phase 2: fine-tune the entire network."""
        for param in self.model.parameters():
            param.requires_grad = True


def make_model(cfg, num_classes: int) -> RisClassifier:
    return RisClassifier(
        num_classes=num_classes,
        dropout=cfg["model"].get("dropout", 0.3),
        arch=cfg["model"].get("arch", "efficientnet_b2"),
    )
