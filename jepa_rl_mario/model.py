import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from functools import reduce

class MarioModel(nn.Module):

    _num_actions: typing.Final[int]
    _width: typing.Final[int]
    _height: typing.Final[int]
    _STRIDES: typing.Final[list[int]] = [4, 2]

    def __init__(self, height, width, num_actions: int):
        """
        Initialize the MarioModel.

        Args:
            num_actions (int): The number of possible actions in the environment.
            width (int): The width of the input image.
            height (int): The height of the input image.

        The image is expected to be height-major, with the images having shape (height, width, channels).
        """
        super(MarioModel, self).__init__()
        self._num_actions = num_actions
        self._width = width
        self._height = height
        final_channels = 32
        self._conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=self._STRIDES[4]),
            nn.ReLU(),
            nn.Conv2d(16, final_channels, kernel_size=4, stride=self._STRIDES[2]),
            nn.ReLU(),
        )
        total_reduction: int = 1
        for i in self._STRIDES:
            total_reduction *= i
        fc_input = int(final_channels * height / total_reduction * width / total_reduction)
        self._fc = nn.Sequential(
            nn.Linear(fc_input, 256),
            nn.ReLU(),
            nn.Linear(256, self._num_actions)
        )

    def forward(self, x):
        conv_out = self._conv(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self._fc(conv_out)


class DownScaleImage(nn.Module):
    """
    Downscale an image and its channel
    """

    def __init__(self, width: int, height: int):
        super(DownScaleImage, self).__init__()
        self.width = width
        self.height = height


    def forward(self, x):
        pass