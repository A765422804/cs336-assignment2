import torch
from torch import nn
import torch.distributed as dist

class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        '''
        作为module的上层DDP的wrapper类
        '''
        super().__init__()
        for parameter in module.parameters():
            with torch.no_grad():
                dist.broadcast(parameter, 0)
        self.module = module

    def forward(self, x):
        return self.module(x)
    
    def finish_gradient_sync(self):
        '''
        backward 之后 step 之前，把所有的device的梯度平均
        '''
        world_size = dist.get_world_size()
        with torch.no_grad():
            for parameter in self.module.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, async_op=False)
                    parameter.grad /= world_size
