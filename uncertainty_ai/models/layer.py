import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter, Identity, Dropout
import torch.utils.data as data
from torch.autograd import Variable

from torch_geometric.nn.inits import glorot_orthogonal
"""https://github.com/j-min/Dropouts/blob/master/Gaussian_Variational_Dropout.ipynb

Returns:
    _type_: _description_
"""

class VariationalDropout(nn.Module):
    def __init__(self, alpha=1.0, dim=None):
        super(VariationalDropout, self).__init__()
        
        self.dim = dim
        self.max_alpha = alpha
        # Initial alpha
        log_alpha = (torch.ones(dim) * alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)
        
    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921
        
        alpha = self.log_alpha.exp()
        
        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha**2 + c3 * alpha**3
        
        kl = -negative_kl
        
        return kl.mean()
    
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(0,1)
            epsilon = Variable(torch.randn(x.size()))
            if x.is_cuda:
                epsilon = epsilon.cuda()

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
            alpha = self.log_alpha.exp()

            # N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon
        else:
            return x

class GaussianDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()
        assert 0 < p < 1
        self.p = p
        
    def forward(self, x):
        if self.training:
            mean = torch.ones_like(x)
            stddev = (self.p / (1.0 - self.p))**0.5
            epsilon = torch.normal(mean = mean, std = stddev) 
            return x * epsilon
        else:
            return x
        
class EquivariantVectorDropout(torch.nn.Module):
    def __init__(self, dropout_method = "bernoulli", p=0.5):
        super(EquivariantVectorDropout, self).__init__()
        assert 0 < p < 1
        self.p = p
        if dropout_method == "bernoulli":
            self.dropout = Dropout(p)
        elif dropout_method == "gaussian":
            self.dropout = GaussianDropout(p)
        else:
            raise ValueError("Unknown dropout method: {}".format(dropout_method))

        
    def forward(self, x):
        assert x.size(1) == 3
        if self.training:
            mask = torch.ones(x.size(0), x.size(-1), device = x.device)
            mask = self.dropout(mask)
            mask = mask.unsqueeze(1)
            return x * mask
        else:
            return x


class SBPDropout(nn.Module):
    """
    Structured Bayesian Pruning layer

    #don't forget add kl to loss
    y, kl = sbp_layer(x)
    loss = loss + kl
    """
    def __init__(self, dim=None, z_channel = 1, init_logsigma2=9.0):
        super(SBPDropout, self).__init__()
        self.input_dim = dim
        self.z_channel = z_channel
        self.init_logsigma2 = init_logsigma2
        self.log_sigma2 = nn.Parameter(torch.Tensor(dim,1))
        self.mu = nn.Parameter(torch.Tensor(dim,1))
        self.mu.data.normal_(1.0, 0.01)
        self.log_sigma2.data.fill_(-init_logsigma2)
        self.sigmoid = nn.Sigmoid()
    
    def reset_parameters(self):
        self.log_sigma2.data.fill_(-self.init_logsigma2)
        self.mu.data.normal_(1.0, 0.01)
 
    def forward(self, input):
        self.log_alpha = self.log_sigma2 - 2.0 * torch.log(abs(self.mu) + 1e-7)
        self.log_alpha = torch.clamp(self.log_alpha, -10.0, 10.0)
 
        if self.training:
            si = (self.log_sigma2).mul(0.5).exp_()
            eps = si.data.new(si.size()).normal_()
            multiplicator =  (self.mu + si*eps)
        else:
            multiplicator =  (self.mu)
 
        expend = multiplicator.expand(self.input_dim, self.z_channel)
        expend = expend.reshape(self.input_dim*self.z_channel)
        return expend*input #multiplicator*input
 
    def kl(self):
        kl = 0.5 * torch.log1p(torch.exp(-self.log_alpha))#*self.z_channel
        a = torch.sum(kl)
        return a

class Linearvbd1(nn.Module):
    def __init__(self,in_features, out_features, threshold=3, bias=True):
        super(Linearvbd1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        self.W = Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()
        glorot_orthogonal(self.W, scale=2.0)
        # self.W.data.normal_(0, 0.001)
        self.log_sigma.data.fill_(-4) # -9

    def forward(self, x):
        self.log_alpha = self.log_sigma * 2.0 - 2.0 * torch.log(1e-8 + torch.abs(self.W))
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10)

        if self.training:
            lrt_mean = nn.functional.linear(x, self.W) + self.bias
            lrt_std = torch.sqrt(nn.functional.linear(x * x, torch.exp(self.log_sigma * 2.0)) + 1e-8)
            eps = lrt_std.data.new(lrt_std.size()).normal_()
            return lrt_mean + lrt_std * eps
        mask = (self.log_alpha < self.threshold).float()
        #return F.linear(x, self.W * mask) + self.bias
        return nn.functional.linear(x, self.W) + self.bias

    def kl(self):
        kl = 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        a = torch.sum(kl)
        return a
    

class EquivariantVectorVBDDropout(torch.nn.Module):
    def __init__(self, vbd_method, dim):
        super(EquivariantVectorVBDDropout, self).__init__()
        if vbd_method == "sbp":
            self.dropout = SBPDropout(dim=dim)
        elif vbd_method == "variational":
            self.dropout = VariationalDropout(dim=dim)
        else:
            raise ValueError("Unknown dropout method: {}".format(vbd_method))

    def forward(self, x):
        assert x.size(1) == 3
        if self.training:
            mask = torch.ones(x.size(0), x.size(-1), device = x.device)
            mask = self.dropout(mask)
            mask = mask.unsqueeze(1)
            return x * mask
        else:
            return x
    
    def kl(self):
        return self.dropout.kl()


class NodeDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(NodeDropout, self).__init__()
        assert 0 < p < 1
        self.p = p
        
    def forward(self, x):
        if self.training:
            mean = torch.ones(x.size(0), 1, device = x.device)- self.p
            epsilon = torch.bernoulli(mean)
            return x * epsilon
        else:
            return x
        
class NodeGaussianDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(NodeGaussianDropout, self).__init__()
        assert 0 < p < 1
        self.p = p
        
    def forward(self, x):
        if self.training:
            mean = torch.ones(x.size(0), 1, device = x.device)
            stddev = (self.p / (1.0 - self.p))**0.5
            epsilon = torch.normal(mean = mean, std = stddev) 
            return x * epsilon
        else:
            return x


class EdgeDropout(nn.Module):
    def __init__(self, p=0.5):
        super(EdgeDropout, self).__init__()
        self.p = p

    def forward(self, edge_index, edge_attr, edge_weight=None):
        if self.training:
            # Generate a mask for edges
            mask = torch.rand(edge_index.size(1)) > self.p
            # Select edges based on the mask
            edge_index = edge_index[:, mask]
            edge_attr = edge_attr[mask]
            if edge_weight is not None:
                edge_weight = edge_weight[mask]
        if edge_weight is not None:
            return edge_index, edge_attr, edge_weight
        else:
            return edge_index, edge_attr

class EdgeGaussianDropout(nn.Module):
    def __init__(self, p, rbf=Identity()):
        super(EdgeGaussianDropout, self).__init__()
        assert 0 < p < 1
        self.p = p
        self.distance_expansion = rbf

    def forward(self, edge_index, edge_attr, edge_weight):
        if self.training:
            mean = torch.ones_like(edge_weight)
            stddev = (self.p / (1.0 - self.p))**0.5
            epsilon = torch.normal(mean = mean, std = stddev)
            new_edge_weight = edge_weight * epsilon
            new_edge_weight = torch.clamp(new_edge_weight, 
                                            self.distance_expansion.offset[0],
                                            self.distance_expansion.offset[-1])
            new_edge_attr = self.distance_expansion(new_edge_weight)
            return edge_index, new_edge_attr, new_edge_weight
        else:
            return edge_index, edge_attr, edge_weight


class EdgeGaussianNoise(nn.Module):
    def __init__(self, p, rbf=Identity()):
        super(EdgeGaussianNoise, self).__init__()
        assert 0 < p < 1
        self.p = p
        self.distance_expansion = rbf

    def forward(self, edge_index, edge_attr, edge_weight):
        if self.training:
            mean = torch.ones_like(edge_weight)
            stddev = (self.p / (1.0 - self.p))**0.5
            epsilon = torch.normal(mean = mean, std = stddev)
            new_edge_weight = edge_weight + epsilon - 1
            # only do the following if the selfdistance_expansion is not Identity
            if not isinstance(self.distance_expansion, Identity):
                new_edge_weight = torch.clamp(new_edge_weight, 
                                                self.distance_expansion.offset[0],
                                                self.distance_expansion.offset[-1])
            new_edge_attr = self.distance_expansion(new_edge_weight)
            return edge_index, new_edge_attr, new_edge_weight
        else:
            return edge_index, edge_attr, edge_weight


class EquivariantVectorSBPDropout(torch.nn.Module):
    """
    Equivariant Vector Structured Bayesian Pruning Dropout
    
    Applies SBP dropout to equivariant vector features (3D vectors) while maintaining
    equivariance by applying the same mask across all vector components.
    
    #don't forget add kl to loss
    y, kl = equivariant_sbp_layer(x)
    loss = loss + kl
    """
    def __init__(self, dim, init_logsigma2=9.0):
        super(EquivariantVectorSBPDropout, self).__init__()
        self.dim = dim
        self.init_logsigma2 = init_logsigma2
        
        # SBP parameters for each dimension
        self.log_sigma2 = nn.Parameter(torch.Tensor(dim, 1))
        self.mu = nn.Parameter(torch.Tensor(dim, 1))
        
        # Initialize parameters
        self.mu.data.normal_(1.0, 0.01)
        self.log_sigma2.data.fill_(-init_logsigma2)
    
    def reset_parameters(self):
        """Reset parameters to initial values"""
        self.log_sigma2.data.fill_(-self.init_logsigma2)
        self.mu.data.normal_(1.0, 0.01)
    
    def forward(self, x):
        """
        Forward pass for equivariant vector SBP dropout
        
        Args:
            x: Input tensor of shape (batch_size, 3, feature_dim) for 3D vectors
            
        Returns:
            Masked tensor of same shape as input
        """
        assert x.size(1) == 3, f"Expected 3D vectors, got shape {x.shape}"
        
        # Compute log_alpha for SBP
        self.log_alpha = self.log_sigma2 - 2.0 * torch.log(abs(self.mu) + 1e-7)
        self.log_alpha = torch.clamp(self.log_alpha, -10.0, 10.0)
        
        if self.training:
            # Sample from Gaussian distribution
            si = (self.log_sigma2).mul(0.5).exp_()
            eps = si.data.new(si.size()).normal_()
            multiplicator = (self.mu + si * eps)
        else:
            # Use mean during inference
            multiplicator = self.mu
        
        # Reshape multiplicator to match input dimensions
        # multiplicator shape: (dim, 1) -> (1, 1, dim)
        mask = multiplicator.reshape(1, 1, self.dim)
        
        # Apply mask to all vector components (maintaining equivariance)
        return x * mask
    
    def kl(self):
        """
        Compute KL divergence for variational inference
        
        Returns:
            KL divergence term to be added to the loss
        """
        kl = 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        return torch.sum(kl)

