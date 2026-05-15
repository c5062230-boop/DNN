import torch
import torch.nn as nn
import torchvision.models as models


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()

        # CNN Encoder (ResNet18 pretrained)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        self.feature_dim = 512

        # LSTM Sequence Model
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # Output layer (IMPORTANT for marks)
        self.fc = nn.Linear(512, 256)

    def forward(self, x):
        # x: [B, S, 3, 224, 224]
        B, S, C, H, W = x.shape

        # reshape for CNN
        x = x.view(B * S, C, H, W)

        # CNN features
        features = self.cnn(x)             # [B*S, 512, 1, 1]
        features = features.view(B, S, -1) # [B, S, 512]

        # LSTM
        lstm_out, _ = self.lstm(features)  # [B, S, 512]

        # Fully connected
        output = self.fc(lstm_out)         # [B, S, 256]

        return output