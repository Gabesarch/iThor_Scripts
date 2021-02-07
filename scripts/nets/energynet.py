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
import utils.misc

class Context2dNet(nn.Module):
    def __init__(self, app_emb_dim, n_head, n_hidden, n_layers, dropout):
        """
        Args:
            app_emb_dim: dimension of appearance embedding
            n_head: number of attention heads
            h_hidden: dimension of the feedforward network
            n_layers: number of layers in the transformer model
            dropout: probability of dropping out
        """
        super(Context2dNet, self).__init__()
        self.app_emb_dim = app_emb_dim
        self.n_head = n_head
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout

        # ResNet-18 base architecture CNN
        self.object_cnn = models.resnet18()
        self.object_cnn.layer4 = nn.Identity()
        self.object_cnn.fc = nn.Linear(in_features=256, out_features=app_emb_dim, bias=True)

        # Mask token
        self.mask_token = nn.Embedding(1, app_emb_dim)

        # Transformer
        encoder_layers = TransformerEncoderLayer(app_emb_dim, n_head, n_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        # Linear layer after final transformer output
        self.app_decoder = nn.Linear(app_emb_dim, app_emb_dim)

        self.init_weights()

    def init_weights(self):
        self.mask_token.weight.data.uniform_(-1.0, 1.0)
        self.app_decoder.bias.data.zero_()
        self.app_decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, obj_crop, pos_emb, pad_mask, mask_mask):
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
        N, S, C, H, W = obj_crop.shape

        # Get object embeddings from crops
        obj_emb = self.object_cnn(obj_crop.reshape(N*S, C, H, W)) # (N*S, E)
        obj_emb = obj_emb.reshape(N, S, self.app_emb_dim).permute(1,0,2) # (S, N, E)
        cnn_emb = obj_emb.clone()

        # Mask out objects, 1 = mask token, 2 = random token
        obj_emb[mask_mask.permute(1,0) == 1,:] = self.mask_token(torch.LongTensor([0]).cuda())[0]
        rand_n = obj_emb[mask_mask.permute(1,0) == 2,:].shape[0]
        obj_emb[mask_mask.permute(1,0) == 2,:] = torch.rand((rand_n, self.app_emb_dim)).cuda() - 0.5

        # Pos encode
        obj_emb += pos_emb

        tf_emb = self.transformer_encoder(obj_emb, src_key_padding_mask=pad_mask)
        tf_emb = self.app_decoder(tf_emb)

        st()

        return cnn_emb, tf_emb