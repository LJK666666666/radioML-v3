"""
SNR Predictor models (PyTorch).

Treats discrete SNR values (-20 to 18 dB, step 2) as a 20-class classification task.
Registry pattern allows easy addition of new predictor architectures.
"""

import torch
import torch.nn as nn


class SNRPredictorCNN(nn.Module):
    """CNN-based SNR predictor for I/Q signals.

    Input: (B, 2, seq_len)  -- I/Q channels
    Output: (B, num_snr_classes) logits
    """

    def __init__(self, input_channels=2, seq_len=128, num_snr_classes=20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_snr_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)  # (B, 256)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

SNR_MODEL_BUILDERS = {
    'snr_cnn': SNRPredictorCNN,
}


def build_snr_predictor(model_name, input_channels=2, seq_len=128, num_snr_classes=20):
    """Build an SNR predictor model by name.

    Args:
        model_name: Key in SNR_MODEL_BUILDERS (e.g., 'snr_cnn')
        input_channels: Number of input channels (default 2 for I/Q)
        seq_len: Sequence length (default 128)
        num_snr_classes: Number of discrete SNR classes (default 20)

    Returns:
        nn.Module instance
    """
    if model_name not in SNR_MODEL_BUILDERS:
        raise ValueError(f"Unknown SNR model: {model_name}. "
                         f"Available: {list(SNR_MODEL_BUILDERS.keys())}")
    cls = SNR_MODEL_BUILDERS[model_name]
    return cls(input_channels=input_channels, seq_len=seq_len, num_snr_classes=num_snr_classes)


def get_available_snr_models():
    """Return list of available SNR predictor model names."""
    return list(SNR_MODEL_BUILDERS.keys())
