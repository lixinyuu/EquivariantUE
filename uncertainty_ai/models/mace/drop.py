'''
    Add `extra_repr` into DropPath implemented by timm 
    for displaying more info.
'''


import torch
import torch.nn as nn
from e3nn import o3
import torch.nn.functional as F
from ..layer import SBPDropout
    

class EquivariantDropout(nn.Module):
    def __init__(self, irreps, drop_prob):
        super(EquivariantDropout, self).__init__()
        self.irreps = irreps
        self.num_irreps = irreps.num_irreps
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)
        self.mul = o3.ElementwiseTensorProduct(irreps, 
            o3.Irreps('{}x0e'.format(self.num_irreps)))
        
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = (x.shape[0], self.num_irreps)
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        out = self.mul(x, mask)
        return out


class EquivariantSBPDropout(nn.Module):
    def __init__(self, irreps):
        super(EquivariantSBPDropout, self).__init__()
        self.irreps = irreps
        self.num_irreps = irreps.num_irreps
        self.mul = o3.ElementwiseTensorProduct(irreps, 
            o3.Irreps('{}x0e'.format(self.num_irreps)))
        self.drop = SBPDropout(dim=self.num_irreps)
        
    def forward(self, x):
        if not self.training:
            return x
        shape = (x.shape[0], self.num_irreps)
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        out = self.mul(x, mask)
        return out
    
    def kl(self):
        """Return KL divergence of the dropout distribution."""
        return self.drop.kl()
    

class EquivariantScalarsDropout(nn.Module):
    def __init__(self, irreps, drop_prob):
        super(EquivariantScalarsDropout, self).__init__()
        self.irreps = irreps
        self.drop_prob = drop_prob
        
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        out = []
        start_idx = 0
        for mul, ir in self.irreps:
            temp = x.narrow(-1, start_idx, mul * ir.dim)
            start_idx += mul * ir.dim
            if ir.is_scalar():
                temp = F.dropout(temp, p=self.drop_prob, training=self.training)
            out.append(temp)
        out = torch.cat(out, dim=-1)
        return out