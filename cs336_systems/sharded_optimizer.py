import torch
from torch.optim import Optimizer
import torch.distributed as dist
from typing import Type,Any
import math

class ShardedOptimizer(torch.optim.Optimizer):
    '''
    optimizer 的 wrapper，实现state的分rank存储
    '''
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        '''
        按照rank对参数划分：根据idx % world_size来决定在哪个rank里面
        1. 调用父类构造函数，父类构造函数会调用add_param_group来初始化param -> rank
        2. 基于分好片的groups初始化optimizer
        '''

        self.param_idx = 0
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.cur_rank_groups = []
        self.is_inited = False

        super().__init__(params, kwargs)

        self.optimizer = optimizer_cls(self.cur_rank_groups, **kwargs)

        self.is_inited = True


    def step(self, closure = None, **kwargs):
        loss = self.optimizer.step(closure, **kwargs)

        '''
        广播当前的参数
        '''
        with torch.no_grad():
            idx = 0
            for group in self.param_groups:
                for param in group['params']:
                    dist.broadcast(param, idx % self.world_size)
                    idx += 1

        return loss
            
    def add_param_group(self, param_group: dict[str, Any]):
        '''
        基于param_idx来决定当前group中参数是否属于当前rank
        '''
        group_copy = param_group.copy()
        group_copy['params'] = list(group_copy['params'])

        super().add_param_group(group_copy)

        cur_rank_params = []
        for param in group_copy['params']:
            if self.param_idx % self.world_size == self.rank:
                cur_rank_params.append(param)
            self.param_idx += 1
        if len(cur_rank_params) > 0:
            cur_group = group_copy.copy()
            cur_group['params'] = cur_rank_params
            self.cur_rank_groups.append(cur_group)
            if self.is_inited:
                self.optimizer.add_param_group(cur_group)
