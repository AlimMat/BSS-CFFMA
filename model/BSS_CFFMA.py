import torch, pdb
import torch.nn as nn
from model.WavLM import WavLM, WavLMConfig 
import fairseq
from utils.util import get_feature, feature_to_wav
from model.MSCFF import MSCFF
from model.encoder import RHMAblock

class BSS_CFFMA(nn.Module):
    
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.transform = get_feature()
        if not args.feature=='raw':
            self.dim = 768 if 'base' in args.size else 1024
            weight_dim = 13 if 'base' in args.size else 25
            if not args.ssl_model=='wavlm':
                weight_dim = weight_dim-1
            if args.weighted_sum:
                self.weights = nn.Parameter(torch.ones(weight_dim))
                self.softmax = nn.Softmax(-1)
                layer_norm  = []
                for _ in range(weight_dim):
                    layer_norm.append(nn.LayerNorm(self.dim))
                self.layer_norm = nn.Sequential(*layer_norm)
            
        if args.feature=='raw':
            embed = 201
        elif args.feature=='ssl':
            embed = self.dim
        else:
            embed = 201+self.dim 
        self.smffb = MFFB(201,self.dim,embed,960)
        self.rhma = nn.Sequential(
            nn.Linear(960, 512, bias=True),
            sqblock(encoder_dim=512,num_layers=2),
            nn.Linear(512, 201, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self,wav,layer_reps,output_wav=False,layer_norm=True,output_tesn=False):
        
        (log1p,_phase),_len = self.transform(wav,ftype='log1p')
        if self.args.feature!='raw':
            if self.args.weighted_sum:
                ssl = torch.cat(layer_reps,2)
            else:
                ssl = layer_reps[-1]
            B,T,embed_dim = ssl.shape
            ssl = ssl.repeat(1,1,2).reshape(B,-1,embed_dim)
            ssl = torch.cat((ssl,ssl[:,-1:].repeat(1,log1p.shape[2]-ssl.shape[1],1)),dim=1)
            if self.args.weighted_sum:
                lms  = torch.split(ssl, self.dim, dim=2)
                for i,(lm,layer,weight) in enumerate(zip(lms,self.layer_norm,self.softmax(self.weights))):
                    if layer_norm:
                        lm = layer(lm)
                    if i==0:
                        # print('layer {} weight: {}'.format(i,weight))
                        out = lm*weight
                    else:
                        # print('layer {} weight: {}'.format(i,weight))
                        out = out+lm*weight
                x    = torch.cat((log1p.transpose(1,2),out),2) if self.args.feature=='cross' else out 
            else:
                x    = torch.cat((log1p.transpose(1,2),ssl),2) if self.args.feature=='cross' else ssl 
        else:
            x = log1p.transpose(1,2)
        
        out1 = self.smffb(log1p,out.transpose(1,2),x.transpose(1,2))
        a = self.lstm_enc[0](out1.transpose(1,2))
        b = self.lstm_enc[1](a)
        out2 = self.lstm_enc(out1.transpose(1,2)).transpose(1,2)
        if self.args.target=='IRM':
            out = out2*log1p
        if output_wav:
            out = feature_to_wav((out,_phase),_len)
        if output_tesn:
            return b,log1p
        return out
        
    
def MainModel(args):
    
    model = BSS_CFFMA(args)
    
    return model