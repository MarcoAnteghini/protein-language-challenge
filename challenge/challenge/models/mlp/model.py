import torch
import torch.nn as nn
import numpy as np

from challenge.base import ModelBase
from challenge.utils import setup_logger


log = setup_logger(__name__)


class MLP(ModelBase):
    def __init__(self, in_features: int):
        """ Simple baseline model for prediction secondary structure
        Args:
            in_features: size in features
        """
        super(MLP, self).__init__()

        # Task block
        self.linear1 = nn.Linear(in_features=in_features, out_features=2048)
        self.linear2 = nn.Linear(in_features=2048, out_features=512)
        self.linear3 = nn.Linear(in_features=512, out_features=128)
        self.linear4 = nn.Linear(in_features=128, out_features=32)
        self.ss8 = nn.Linear(in_features=32, out_features=8)

        log.info(f'<init>: \n{self}')

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic """
        
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.linear3(x)
        x = nn.ReLU()(x)
        x = self.linear4(x)
        x = nn.ReLU()(x)
        ss8 = self.ss8(x)

        ss8 = nn.functional.softmax(ss8, 2)
        ss3 = torch.stack([torch.sum(ss8[:, :, :3], 2),\
                           torch.sum(ss8[:, :, 3:5], 2),\
                           torch.sum(ss8[:, :, 5:], 2)], dim=2)

        return [ss8, ss3]
