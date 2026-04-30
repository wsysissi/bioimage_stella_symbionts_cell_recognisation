import torch
import torch.nn as nn


class CellDINOClassificationHead(nn.Module):
    def __init__(self, _input_dim=1024, num_classes=2, dropout=0.2):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """
        x shape: [batch_size, 1024]
        output shape: [batch_size, 3]
        """
        return self.classifier(x)