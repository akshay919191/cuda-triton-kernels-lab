import torch
import torch.nn as nn
import torch.nn.functional as F

def reduce(listt):
    total = 1
    for x in listt:
        total *= x
    return total

class RNSnorm(nn.Module):
    def __init__(self , dim , eps = None):
        super().__init__()
        self.dim   = dim
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(
            torch.mean(x * x, dim=-1, keepdim=True)
            + self.eps
        )

        x = x / rms
        x = x * self.gamma

        return x
    
    @classmethod
    def make_class(cls , dim , eps):
        return cls(dim , eps)

eps = 1e-3
rns = RNSnorm.make_class(1024, eps).cuda()
rnstorch = nn.RMSNorm(1024, eps).cuda()

## test
input_t = torch.randn((1024 , 1024) , device = 'cuda')

a = rns(input_t)
b = rnstorch(input_t)

print(torch.allclose(a , b , atol = 1e-2))
print((a - b).abs().max())