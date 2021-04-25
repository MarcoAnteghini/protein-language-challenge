import torch
import torch.nn as nn
import numpy as np

from challenge.base import ModelBase
from challenge.utils import setup_logger


log = setup_logger(__name__)


class CNN1D(ModelBase):
    def __init__(self, in_features: int):
        """ Simple baseline model for prediction secondary structure
        Args:
            in_features: size in features
        """
        super(CNN1D, self).__init__()

        # Task block
        #self.cnn1 = nn.Conv1d(in_channels=1280, out_channels=1280*2, kernel_size=3, padding=1)
        #self.cnn2 = nn.Conv1d(in_channels=1280*2, out_channels=1280*4, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(input_size=1280, hidden_size=256, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(in_features=256*2, out_features=64)
        #self.linear4 = nn.Linear(in_features=128, out_features=32)
        self.ss8 = nn.Linear(in_features=64, out_features=8)

        log.info(f'<init>: \n{self}')

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic """
        # 16, 1280, 1632
        #x = x.permute(0, 2, 1)        

        #x = self.cnn1(x)
        #x = nn.ReLU()(x)
        #x = self.pool(x)
        #x = self.cnn2(x)
        #x = nn.ReLU()(x)

        #x = x.permute(0, 2, 1)
        x, (h, c) = self.bilstm(x)
        x = self.linear(x)
        
        ss8 = self.ss8(x)
        ss8 = nn.functional.softmax(ss8, 2)
        ss3 = torch.stack([torch.sum(ss8[:, :, :3], 2),\
                           torch.sum(ss8[:, :, 3:5], 2),\
                           torch.sum(ss8[:, :, 5:], 2)], dim=2)

        return [ss8, ss3]
