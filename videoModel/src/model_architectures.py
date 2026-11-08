"""Classifier architectures shared by training and inference."""

import torch
from torch import nn

try:
    import torchvision
except ImportError:  # pragma: no cover
    torchvision = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _mobilenet_v3_small_features(pretrained: bool) -> nn.Module:
    """MobileNetV3-Small feature trunk; pretrained weights download on demand."""
    if torchvision is None:
        raise RuntimeError("torchvision is required for the MobileNetV3 frame encoder.")
    try:
        return torchvision.models.get_model(
            "mobilenet_v3_small",
            weights="IMAGENET1K_V1" if pretrained else None,
        ).features
    except Exception as error:
        if not pretrained:
            raise
        print(
            f"WARNING: pretrained MobileNetV3 weights could not be loaded ({error}); "
            "the frame encoder will be trained from scratch."
        )
        try:
            return torchvision.models.get_model("mobilenet_v3_small", weights=None).features
        except Exception:
            return torchvision.models.mobilenet_v3_small(weights=None).features


class FrameCNNGRU(nn.Module):
    """Pretrained MobileNetV3 frame encoder, bidirectional GRU, temporal attention."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = _mobilenet_v3_small_features(pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gru = nn.GRU(576, 256, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.LayerNorm(512), nn.Dropout(0.4), nn.Linear(512, num_classes))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1))

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        batch, frames, channels, height, width = videos.shape
        normalized = (videos - self.mean) / self.std
        features = self.backbone(normalized.reshape(batch * frames, channels, height, width))
        features = self.pool(features).flatten(1).reshape(batch, frames, -1)
        sequence, _ = self.gru(features)
        scores = self.attention(sequence).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = (sequence * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(context)


class ResidualBlock3D(nn.Module):
    """Bottleneck-free 3D residual block with an optional strided shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class VideoCNN3D(nn.Module):
    """Residual 3D CNN over spatial and temporal dimensions."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
            ResidualBlock3D(32, 32),
            ResidualBlock3D(32, 32),
            ResidualBlock3D(32, 64, stride=(2, 2, 2)),
            ResidualBlock3D(64, 64),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(videos.permute(0, 2, 1, 3, 4)))


def build_model(name: str, class_count: int, pretrained: bool = True) -> nn.Module:
    if name == "cnn_gru":
        return FrameCNNGRU(class_count, pretrained=pretrained)
    if name == "cnn3d":
        return VideoCNN3D(class_count)
    raise ValueError(f"Unknown model: {name}")
