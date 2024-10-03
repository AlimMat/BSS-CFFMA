#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   SMFFB.py
@Time    :   2024/03/11 17:10:31
@Author  :   AliMu 
@Version :   1.0
@Site    :   2778585742@qq.com
@Desc    :   None
'''
import torch
import torch.nn as nn
from torch.nn import functional as F

def select_norm(norm, dim):
    if norm not in ['gln', 'cln', 'bn']:
        exit(-1)

    if norm == 'gln':
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    else:
        return nn.BatchNorm1d(dim)

class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
            input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
           this module has learnable per-element affine parameters 
           initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
        else:
            x = (x-mean)/torch.sqrt(var+self.eps)
        return x

class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters 
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: N x C x L
        # N x L x C
        x = torch.transpose(x, 1, 2)
        # N x L x C == only channel norm
        x = super().forward(x)
        # N x C x L
        x = torch.transpose(x, 1, 2)
        return x

class Dconv(nn.Module):
            
    def __init__(self,in_channels,out_channels,kernel_size=3,pading=1,stride=1, dilation=1,causal=False) -> None:
        super(Dconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        # self.pad = (self.dilation * (self.kernel_size - 1)) // 2 if not causal else (dilation * (kernel_size - 1))
        # depthwise convolution
        self.dwconv = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=kernel_size,padding=pading,stride=stride,dilation=dilation)

    def forward(self,x):
        x = self.dwconv(x)
        return x

class MSCFF(nn.Module):

    def __init__(self, in_channels_stft,in_channels_ssl,in_channels_ff,out_channels,kernel_size=3, dilation=1, norm='gln', causal=False,):
        super(MSCFF, self).__init__()
        # Hyper-parameter
        self.in_channels_stft = in_channels_stft
        self.in_channels_ssl = in_channels_ssl
        self.in_channels_ff = in_channels_ff
        self.out_channels_stft = in_channels_stft
        self.out_channels_ssl = in_channels_ssl
        self.out_channels_ff = in_channels_ff
        self.in_channels = out_channels
        self.out_channels = out_channels

        self.liner = nn.Linear(self.in_channels_stft+self.in_channels_ssl,self.out_channels)

        self.conv1d_1 = nn.Conv1d(self.in_channels, int(self.out_channels/2), kernel_size=1, bias=False)
        self.sigmod_1 = nn.Sigmoid()
        self.prelu = nn.PReLU()
        self.norm = select_norm('gln', int(self.out_channels/2))
        self.relu = nn.ReLU()

        self.dconv_stft = Dconv(int(self.out_channels/2),self.out_channels_stft,kernel_size=3, pading=1,stride=1,dilation=1)
        self.dconv_ssl = Dconv(int(self.out_channels/2),self.out_channels_ssl,kernel_size=5,pading=2,stride=1, dilation=1)
        self.dconv_ff = Dconv(int(self.out_channels/2),self.out_channels,kernel_size=7,pading=3,stride=1, dilation=1)


    def forward(self, stft_feature, ssl_feature,ssl_stft):
        # print(stft_feature.shape)
        # print(ssl_feature.shape)
        # print(ssl_stft.shape)
        
        fusion_feature_cat_liner = self.liner(ssl_stft.transpose(1,2))
        fusion_feature = self.conv1d_1(fusion_feature_cat_liner.transpose(1,2))
        fusion_feature_prlu = self.prelu(fusion_feature)
        fusion_feature_prlu_norm =self.norm(fusion_feature_prlu)
        # print(fusion_feature_prlu_norm.shape)

        stft_dv_sig = self.sigmod_1(self.dconv_stft(fusion_feature_prlu_norm))
        ssl_dv_sig = self.sigmod_1(self.dconv_ssl(fusion_feature_prlu_norm))
        ff_dv_sig = self.sigmod_1(self.dconv_ff(fusion_feature_prlu_norm))

        stft_out = stft_feature * stft_dv_sig
        ssl_out = ssl_feature * ssl_dv_sig

        ff_out = fusion_feature_cat_liner.transpose(1,2) * ff_dv_sig
        ssl_stft_sig_cat = torch.cat((stft_out,ssl_out),1)
        out = self.relu(self.liner(ssl_stft_sig_cat.transpose(1,2)).transpose(1,2) + ff_out)

        return out
    
    
