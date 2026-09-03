import torch.nn as nn


class GarmentCNN(nn.Module):
    """5-block CNN (conv -> batchnorm -> relu -> pool), GAP, dropout, linear head.

    Same validated architecture as the Sequential version in
    notebooks/03_cnn_from_scratch_secuential.ipynb, rewritten as an
    nn.Module subclass -- see notebooks/03A_cnn_from_scratch_class.ipynb
    for the design rationale (why padding='same', why channels double per
    block, why GAP over a large FC layer, etc).
    """

    def __init__(self, num_classes=14):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.gap(x)
        x = x.flatten(1)   # tensor op, not a layer -- no learnable params, so it doesn't belong in __init__
        x = self.dropout(x)
        x = self.classifier(x)

        return x
