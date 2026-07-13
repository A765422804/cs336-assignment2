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
        grad_list = []
        
        with torch.no_grad():
            for parameter in self.module.parameters():
                if parameter.grad is not None:
                    grad_list.append(parameter.grad)
            
            flat_grad = torch._utils._flatten_dense_tensors(grad_list)
            dist.all_reduce(flat_grad, async_op=False)
            flat_grad /= world_size
            reduced_grad_list = torch._utils._unflatten_dense_tensors(flat_grad, grad_list)

            grad_idx = 0
            for parameter in self.module.parameters():
                if parameter.grad is not None:
                    parameter.grad = reduced_grad_list[grad_idx]
                    grad_idx += 1
