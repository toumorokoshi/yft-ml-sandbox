import torch
import torch.nn as nn

# Constants
BATCH_SIZE = 16
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 400
NUM_CLASSIFICATIONS = 10
SIZE = "320px"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        conv_features = [96, 256, 384, 384, 256]
        conv_output_size = 38400  # with maxpool * 2

        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, conv_features[0], kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(conv_features[0], conv_features[1], kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(conv_features[1], conv_features[2], kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(conv_features[2], conv_features[3], kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(conv_features[3], conv_features[4], kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(conv_output_size, 4096),
            # index 3
            nn.ReLU(),
            nn.Dropout(0.5),
            # index 5
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, NUM_CLASSIFICATIONS),
        )

    def forward(self, x):
        conv_output = self.conv_stack(x)
        logits = self.linear_stack(conv_output)
        return logits
