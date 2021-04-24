import torch
import torch.nn as nn
import torch.nn.functional as F

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
        ss8 = nn.functional.softmax(ss8, 2)
        ss3 = torch.stack([torch.sum(ss8[:, :, :3], 2),\
                           torch.sum(ss8[:, :, 3:5], 2),\
                           torch.sum(ss8[:, :, 5:], 2)], dim=2)

        return [ss8, ss3]
      
class NetSurfModel(ModelBase):
    def __init__(self, in_features: int, hidden_size, lstm_layers, dropout):
        """ Simple implementation of NetSurf model as described in Klausen et al.
        Args:
            in_features: size in features
        """
        super(NetSurfModel, self).__init__()

        self.in_features = in_features
        self.cnn1 = nn.Conv1d(in_channels=in_features, out_channels=int(in_features/32), kernel_size=9, stride=1, padding=4) # in = 1280, out = 40
        self.cnn2 = nn.Conv1d(in_channels=in_features, out_channels=int(in_features/32), kernel_size=9, stride=1, padding=4) # in = 1632, out = 51
        self.bilstm = nn.LSTM(input_size=int(in_features/16), hidden_size=hidden_size, num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.fc1 = nn.Linear(in_features=int(hidden_size*2), out_features=8)
        self.dropout = nn.Dropout(dropout)

        log.info(f'<init>: \n{self}')

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic """
        x = x.permute(0,2,1)

         # Pass embeddings to two parallel CNNs
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)

        # Concatenate outputs of parallel CNNs to form identity layer
        x = torch.cat((x1, x2), dim=1) 

        # Pass identity layer output to two-layer biLSTM
        x = x.permute(0, 2, 1)
        x, (h, c) = self.bilstm(x) # Output of LSTM is output, (hidden, cells)

        # Pass through fully connected layer with 8 outputs
        ss8 = F.softmax(self.fc1(x))

        ss3 = torch.stack([torch.sum(ss8[:, :, :3], 2),\
                           torch.sum(ss8[:, :, 3:5], 2),\
                           torch.sum(ss8[:, :, 5:], 2)], dim=2)

        return [ss8, ss3]

class NetSurfModelA(ModelBase):
    def __init__(self, in_features: int, hidden_size, lstm_layers, dropout):
        """ Implementation of NetSurf model with some freestyling with activation functions and regularization
        Args:
            in_features: size in features
        """
        super(NetSurfModelA, self).__init__()

        self.in_features = in_features
        self.cnn1 = nn.Conv1d(in_channels=in_features, out_channels=int(in_features/32), kernel_size=9, stride=1, padding=4) # in = 1280, out = 40
        self.cnn2 = nn.Conv1d(in_channels=in_features, out_channels=int(in_features/32), kernel_size=9, stride=1, padding=4) # in = 1632, out = 51
        self.bilstm = nn.LSTM(input_size=int(in_features/16), hidden_size=hidden_size, num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.fc1 = nn.Linear(in_features=int(hidden_size*2), out_features=8)
        self.dropout = nn.Dropout(dropout)

        log.info(f'<init>: \n{self}')

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic """
        x = x.permute(0,2,1)

         # Pass embeddings to two parallel CNNs
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)

        # Concatenate outputs of parallel CNNs to form identity layer
        x = torch.cat((x1, x2), dim=1) 

        # Pass identity layer output to two-layer biLSTM
        x = x.permute(0, 2, 1)
        x, (h, c) = self.bilstm(x) # Output of LSTM is output, (hidden, cells)

        # Pass through fully connected layer with 8 outputs
        ss8 = F.softmax(self.fc1(x))

        ss3 = torch.stack([torch.sum(ss8[:, :, :3], 2),\
                           torch.sum(ss8[:, :, 3:5], 2),\
                           torch.sum(ss8[:, :, 5:], 2)], dim=2)

        return [ss8, ss3]

