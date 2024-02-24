import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class ReTanh(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.max(torch.tanh(input), torch.zeros_like(input))


def get_activation_fn(act_fn):
    if isinstance(act_fn, str):
        if act_fn == "ReTanh":
            act_fn = lambda x: torch.max(torch.tanh(x), torch.zeros_like(x))
        else:
            act_fn = getattr(nn, act_fn)()
    return act_fn


# TODO
def computef(IN, string, *args):  # ags[0] is the slope for string='tanhwithslope'
    if string == 'linear':
        F = IN
        return F
    elif string == 'logistic':
        F = 1/(1+torch.exp(-IN))
        return F
    elif string == 'smoothReLU':  # smoothReLU or softplus
        F = torch.log(1+torch.exp(IN))  # always greater than zero
        return F
    elif string == 'ReLU':  # rectified linear units
        # F = torch.maximum(IN,torch.tensor(0))
        F = torch.clamp(IN, min=0)
        return F
    elif string == 'swish':  # swish or SiLU (sigmoid linear unit)
        # Hendrycks and Gimpel 2016 "Gaussian Error Linear Units (GELUs)"
        # Elfwing et al. 2017 "Sigmoid-weighted linear units for neural network function approximation in reinforcement learning"
        # Ramachandran et al. 2017 "Searching for activation functions"
        sigmoid = 1/(1+torch.exp(-IN))
        F = torch.mul(IN, sigmoid)  # x*sigmoid(x), torch.mul performs elementwise multiplication
        return F
    elif string == 'mish':  # Misra 2019 "Mish: A Self Regularized Non-Monotonic Neural Activation Function
        F = torch.mul(IN, torch.tanh(torch.log(1+torch.exp(IN))))  # torch.mul performs elementwise multiplication
        return F
    elif string == 'GELU':  # Hendrycks and Gimpel 2016 "Gaussian Error Linear Units (GELUs)"
        F = 0.5*torch.mul(IN, (1+torch.tanh(
            torch.sqrt(torch.tensor(2/np.pi))*(IN+0.044715*IN**3))))  # fast approximating used in original paper
        # F = x.*normcdf(x,0,1);% x.*normcdf(x,0,1)  =  x*0.5.*(1 + erf(x/sqrt(2)))
        # figure; hold on; x = linspace(-5,5,100); plot(x,x.*normcdf(x,0,1),'k-'); plot(x,0.5*x.*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x.^3))),'r--')
        return F
    elif string == 'ELU':  # exponential linear units, Clevert et al. 2015 "FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS (ELUS)"
        alpha = 1
        inegativeIN = (IN < 0)
        F = IN.clone()
        F[inegativeIN] = alpha*(torch.exp(IN[inegativeIN])-1)
        return F
    elif string == 'Tanh':
        F = torch.tanh(IN)
        return F
    elif string == 'tanhwithslope':
        a = args[0]
        F = torch.tanh(a*IN)  # F(x)=tanh(a*x), dFdx=a-a*(tanh(a*x).^2), d2dFdx=-2*a^2*tanh(a*x)*(1-tanh(a*x).^2)
        return F
    elif string == 'tanhlecun':  # LeCun 1998 "Efficient Backprop"
        F = 1.7159*torch.tanh(
            2/3*IN)  # F(x)=a*tanh(b*x), dFdx=a*b-a*b*(tanh(b*x).^2), d2dFdx=-2*a*b^2*tanh(b*x)*(1-tanh(b*x).^2)
        return F
    elif string == 'lineartanh':
        # F = torch.minimum(torch.maximum(IN,torch.tensor(-1)),torch.tensor(1))# -1(x<-1), x(-1<=x<=1), 1(x>1)
        F = torch.clamp(IN, min=-1, max=1)
        return F
    elif string == 'ReTanh':  # rectified tanh
        F = torch.maximum(torch.tanh(IN), torch.tensor(0))
        return F
    elif string == 'binarymeanzero':  # binary units with output values -1 and +1
        # F = (IN>=0) - (IN<0)# matlab code
        F = 1*(IN >= 0)-1*(IN < 0)  # multiplying by 1 converts True to 1 and False to 0
        return F
    else:
        print('Unknown transfer function type')
