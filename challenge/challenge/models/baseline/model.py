import torch
import torch.nn as nn
import numpy as np

from challenge.base import ModelBase
from challenge.utils import setup_logger


log = setup_logger(__name__)


class Baseline(ModelBase):
    def __init__(self, in_features: int):
        """ Simple baseline model for prediction secondary structure
        Args:
            in_features: size in features
        """
        super(Baseline, self).__init__()

        # Task block
        self.ss8 = nn.Linear(in_features=in_features, out_features=8)
        #self.ss3 = nn.Linear(in_features=in_features, out_features=3)

        log.info(f'<init>: \n{self}')

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic """

        ss8 = self.ss8(x)
        #ss3 = self.ss3(x)
        ss3 = torch.stack([torch.sum(ss8[:, :, :3], 2), torch.sum(ss8[:, :, 3:5], 2), torch.sum(ss8[:, :, 5:], 2)], dim=2)

        return [ss8, ss3]
