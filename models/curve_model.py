from torch import nn
import torch

class curve_model(nn.Module):
    def __init__(self,kernel_size):
        super(curve_model, self).__init__()
        self.layer = nn.ModuleDict({
                'layer0': nn.Sequential(
                    nn.Linear(kernel_size*2,1),
                    )
                })

    def forward(self, sin,cos):
        z = torch.cat((sin,cos))
        for _layer in self.layer.values(): 
              z = _layer(z)
        return z