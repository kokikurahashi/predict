from torch import nn
import torch
class NNmodel(nn.Module):
    def __init__(self,input):
        super(NNmodel, self).__init__()
        self.layer = nn.ModuleDict({
                'layer0': nn.Sequential(
                    nn.Linear(input,int(input/2)),
                     nn.ReLU()
                    ),  
               
                'layer1': nn.Sequential(
                    nn.Linear(int(input/2),int(input/4)),
                    nn.ReLU()
                    ),  
  
                  'layer2': nn.Sequential(
                    nn.Linear(int(input/4),int(input/8)),
                    nn.ReLU()
                    ),  
                    'layer3': nn.Sequential(
                    nn.Linear(int(input/8),1),


                    ),  
  
                    })

    def forward(self, price):
        z = price
 
        for _layer in self.layer.values(): 
              z = _layer(z)

        return z