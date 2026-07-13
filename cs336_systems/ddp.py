import torch
from torch import nn
import torch.distributed as dist

class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        '''
        作为module的上层DDP的wrapper类。
        1. 把rank0的参数广播出去
        2. 初始化参数梯度更新以后的hook函数，从而实现overlap的reduce和backward
        '''
        super().__init__()
        for parameter in module.parameters():
            with torch.no_grad():
                dist.broadcast(parameter, 0)
                if parameter.requires_grad == True:
                    handle = parameter.register_post_accumulate_grad_hook(self.post_acc_grad_hook)
        self.module = module

        self.handle_para_list = []

    def forward(self, x):
        return self.module(x)
    
    def finish_gradient_sync(self):
        '''
        backward 之后 step 之前，把所有的device的梯度平均
        '''
        world_size = dist.get_world_size()
        
        with torch.no_grad():
            for handle, parameter in self.handle_para_list:
                handle.wait()
                parameter.grad /= world_size 
        
        # 清空handle list
        self.handle_para_list = []

    def post_acc_grad_hook(self, parameter: nn.Parameter):
        '''
        hook 函数，在当前参数梯度backward结束时调用
        '''
        with torch.no_grad():
            handle = dist.all_reduce(parameter.grad, async_op=True)
            self.handle_para_list.append((handle, parameter))
