"""Classifier architectures shared by training and inference."""

import torch
from torch import nn


class FrameCNNGRU(nn.Module):
    """Frame-wise spatial CNN followed by a temporal GRU."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = videos.shape
        features = self.encoder(videos.reshape(batch * frames, channels, height, width))
        features = features.flatten(1).reshape(batch, frames, -1)
        sequence, _ = self.gru(features)
        return self.head(sequence[:, -1])


class VideoCNN3D(nn.Module):
    """Compact 3D CNN over spatial and temporal dimensions."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 8, 3, padding=1, bias=False), nn.BatchNorm3d(8), nn.ReLU(inplace=True), nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(8, 16, 3, padding=1, bias=False), nn.BatchNorm3d(16), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1, bias=False), nn.BatchNorm3d(32), nn.ReLU(inplace=True), nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(32, num_classes))

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(videos.permute(0, 2, 1, 3, 4)))


def build_model(name: str, class_count: int) -> nn.Module:
    models = {"cnn_gru": FrameCNNGRU, "cnn3d": VideoCNN3D}
    if name not in models:
        raise ValueError(f"Unknown model: {name}")
    return models[name](class_count)
