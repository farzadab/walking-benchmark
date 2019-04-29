import copy
import torch as th
import torch.nn as nn


class MySequential(nn.Sequential):
    """A more convenient network with some extra functionality"""

    def __init__(self, input_size, output_size, multiplier=1, *args):
        super().__init__(*args)
        self.multiplier = multiplier
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, *args):
        return super().forward(th.cat(args, dim=-1)) * self.multiplier

    def copy(self):
        return copy.deepcopy(self)

    def parameter_moving_average(self, another, tau):
        for p, avg_p in zip(self.parameters(), another.parameters()):
            p.data.mul_(1 - tau).add_(tau, avg_p.data)


def make_net(dims, activations, multiplier=1):
    layers = []
    for i, (inp, out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(inp, out))
        if len(activations) > i:
            layers.append(activations[i])
    return MySequential(dims[0], dims[-1], multiplier, *layers)
