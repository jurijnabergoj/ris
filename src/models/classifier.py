import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class RisClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()

        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Replace the existing 1000-class head with our own
        in_features = self.model.classifier[1].in_features  # 1280 for B0
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

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
    )