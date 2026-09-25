import typing
from typing import Final, Optional
import torch
import torch.nn as nn

# Constants (Rule 5)
DEFAULT_NUM_ACTIONS: Final[int] = 7
DEFAULT_HEIGHT: Final[int] = 80
DEFAULT_WIDTH: Final[int] = 80
DEFAULT_FINAL_CHANNELS: Final[int] = 32
DEFAULT_FC_HIDDEN: Final[int] = 256
STRIDES: Final[list[int]] = [4, 2]


class MarioModel(nn.Module):
    """Deep Q-Network model for Super Mario Bros."""

    _num_actions: Final[int]
    _width: Final[int]
    _height: Final[int]

    def __init__(
        self,
        num_actions: int = DEFAULT_NUM_ACTIONS,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        **kwargs: int,
    ) -> None:
        """
        Initialize the MarioModel.

        Args:
            num_actions (int): The number of possible actions in the environment.
            height (int): The height of the input image.
            width (int): The width of the input image.
        """
        super().__init__()
        # Support positional ordering flexibility if passed as (height, width, num_actions)
        if "num_actions" in kwargs:
            num_actions = kwargs["num_actions"]
        if "height" in kwargs:
            height = kwargs["height"]
        if "width" in kwargs:
            width = kwargs["width"]

        self._num_actions = num_actions
        self._height = height
        self._width = width

        self._conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=STRIDES[0]),
            nn.ReLU(),
            nn.Conv2d(16, DEFAULT_FINAL_CHANNELS, kernel_size=4, stride=STRIDES[1]),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, height, width)
            conv_out = self._conv(dummy)
            fc_input = int(conv_out.numel())

        self._fc = nn.Sequential(
            nn.Linear(fc_input, DEFAULT_FC_HIDDEN),
            nn.ReLU(),
            nn.Linear(DEFAULT_FC_HIDDEN, self._num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self._conv(x)
        flattened = conv_out.view(conv_out.size(0), -1)
        return self._fc(flattened)


class DownScaleImage(nn.Module):
    """Downscale an image and its channel."""

    def __init__(self, width: int, height: int) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x