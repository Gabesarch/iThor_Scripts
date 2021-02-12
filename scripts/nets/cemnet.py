import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import models
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import ipdb
st = ipdb.set_trace

import numpy as np 
import math
import sys
sys.path.append('..')

class CEMNet(nn.Module):
    def __init__(self, h1=32, h2=64, fc_dim=1024, num_actions=4):
        """
        Args:
            app_emb_dim: dimension of appearance embedding
            n_head: number of attention heads
            h_hidden: dimension of the feedforward network
            n_layers: number of layers in the transformer model
            dropout: probability of dropping out
        """
        super(CEMNet, self).__init__()
        self.num_actions = num_actions
        self.h1 = h1
        self.h2 = h2
        self.fc_dim = fc_dim

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.h1, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.h1, out_channels=self.h1, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=self.h1, out_channels=self.h2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.h2, out_channels=self.h2, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_x = nn.Sequential(
            nn.Linear(238144, self.fc_dim),
            nn.ReLU(),
            nn.Linear(self.fc_dim, self.num_actions)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, rgb, softmax=False):
        '''
        Args:
            obj_crop: object rgb crops resized (N, S, C, H, W)
            pos_emb: positional encoding (S, N, E)
            pad_mask: padding mask, (N, S)
            mask_mask: masking mask, (N, S)
        Output:
            cnn_emb: cnn embeddings, (S, N, E)
            tf_emb: transformer embeddings, (S, N, E)
        '''

        out = self.conv_x(rgb)
        out = out.reshape(out.size(0), -1)
        out = self.fc_x(out)

        if softmax:
            out = self.softmax(out)

        return out