import torch
from torch import nn
import torch.nn.functional as F
from .ops import init_network

class ResBlock(nn.Module):
    def __init__(self, in_features, norm=False):
        super(ResBlock, self).__init__()

        block = [  nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3),
                # nn.InstanceNorm2d(in_features),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3),
                # nn.InstanceNorm2d(in_features)
                ]

        if norm:
            block.insert(2,  nn.InstanceNorm2d(in_features))
            block.insert(6,  nn.InstanceNorm2d(in_features))

        self.model = nn.Sequential(*block)
		
    def forward(self, x):
        return x + self.model(x)

class Attn(nn.Module):
    def __init__(self, input_nc=3):
        super(Attn, self).__init__()

        model =  [  nn.Conv2d(3, 32, 7, stride=1, padding=3),
                    nn.InstanceNorm2d(32),
                    nn.ReLU(inplace=True) ]

        model += [  nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        model += [ResBlock(64, norm=True)]

        model += [nn.UpsamplingNearest2d(scale_factor=2)]

        model += [  nn.Conv2d(64, 64, 3, stride=1, padding=1),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]
        
        # model += [nn.UpsamplingNearest2d(scale_factor=2)]

        model += [  nn.Conv2d(64, 32, 3, stride=1, padding=1),
                    nn.InstanceNorm2d(32),
                    nn.ReLU(inplace=True) ]

        model += [  nn.Conv2d(32, 1, 7, stride=1, padding=3),
                    nn.Sigmoid() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

def define_Attn(gpu_ids=[0]):
    attn_net = Attn()
    return init_network(attn_net, gpu_ids)