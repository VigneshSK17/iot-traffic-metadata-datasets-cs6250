"""Train ordered and shuffled-history 1D CNNs on UNSW flows or CIC windows."""

import torch
from torch import nn
from torch.nn import functional as F

from models.lstm import main as run_experiment


CHANNELS = 32
KERNEL_SIZE = 5
DILATIONS = (1, 4)
RECEPTIVE_FIELD = 1 + (KERNEL_SIZE - 1) * sum(DILATIONS)


class FlowCNN(nn.Module):
    def __init__(self, feature_count, class_count):
        super().__init__()
        self.conv1 = nn.Conv1d(feature_count, CHANNELS, KERNEL_SIZE, dilation=DILATIONS[0])
        self.conv2 = nn.Conv1d(CHANNELS, CHANNELS, KERNEL_SIZE, dilation=DILATIONS[1])
        self.head = nn.Linear(CHANNELS, class_count)

    def forward(self, features, lengths):
        states = features.transpose(1, 2)
        for layer in (self.conv1, self.conv2):
            left_padding = (layer.kernel_size[0] - 1) * layer.dilation[0]
            states = layer(F.pad(states, (left_padding, 0))).relu()
        # Read the target timestep, before the right padding in the batch.
        target = states[torch.arange(len(lengths), device=lengths.device), :, lengths - 1]
        return self.head(target)


def main():
    run_experiment(
        model_factory=FlowCNN,
        model_metadata={
            "architecture": "causal_conv1d",
            "channels": CHANNELS,
            "kernel_size": KERNEL_SIZE,
            "dilations": DILATIONS,
            "receptive_field": RECEPTIVE_FIELD,
        },
        description=__doc__,
    )


if __name__ == "__main__":
    main()
