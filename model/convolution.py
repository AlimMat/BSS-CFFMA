# Copyright (c) 2022, Sangchun Ha. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union
import torch
import torch.nn.functional as F

import torch.nn as nn
from torch import Tensor

from model.activation import GLU, Swish
from model.modules import Transpose
from train import get_args


class SCTA(nn.Module):
    """
    Convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by squeezeformer convolution module.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super(SCTA, self).__init__()
        self.SCA = FrequencelAttention(in_channels)
        self.STA = TimeAttentionModule()
        self.relu = nn.ReLU()


    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.SCA(inputs.transpose(1, 2))
        outputs = self.STA(outputs)
        return outputs.transpose(1, 2)


class FrequencelAttention(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.fc = nn.Sequential(nn.Linear(channels, channels//2), nn.ReLU(),
                               nn.Linear(channels//2, channels)) 
        self.relu = nn.PReLU()
        self.a = 0.5       
        
    def forward(self, x):
        """
        Input X: [B,F,T]
        Ouput X: [B,F,T]
        """
        
        # [B,F,T] -> [B,F,1]
        attn_max = F.adaptive_max_pool1d(x, 1)
        attn_avg = F.adaptive_avg_pool1d(x, 1)
        
        attn_max = self.fc(attn_max.squeeze())
        attn_avg = self.fc(attn_avg.squeeze())
        
        # [B,N,1]
        x_max = (self.a/2)*x * F.sigmoid(attn_max).unsqueeze(-1)
        x_avg = (self.a/2)*x * F.sigmoid(attn_avg).unsqueeze(-1)

        attn = attn_max + attn_avg
        attn = F.sigmoid(attn).unsqueeze(-1)
        
        # [B,N,T]
        x_all = (1-self.a)*x * attn
        
        return x_all+x_max+x_avg
    
   
class TimeAttentionModule(nn.Module):
    def __init__(self,):
        super(TimeAttentionModule, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.conv_max_avg = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.a = 0.5
 
    def forward(self, x):
        #map尺寸不变，缩减通道
        # print(x.size())
        # [B,F,T]-->[B,1,T]
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x_max = (self.a/2)*self.sigmoid(self.conv_max_avg(maxout))*x
        x_avg = (self.a/2)*self.sigmoid(self.conv_max_avg(avgout))*x
        out = torch.cat([avgout, maxout], dim=1)
        out = self.conv1d(out)
        out = (1-self.a)*self.sigmoid(out)*x
        return out+x_avg+x_max


class TimeReductionLayer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        stride: int = 2,
    ) -> None:
        super(TimeReductionLayer, self).__init__()
        self.sequential = nn.Sequential(
            DepthwiseConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            Swish(),
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, subsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * subsampled_dim)

        output_lengths = input_lengths >> 1
        output_lengths -= 1
        return outputs, output_lengths
