import os
import timeit
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.ddp import DDP
from cs336_systems.sharded_optimizer import ShardedOptimizer

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('nccl',rank=rank, world_size=world_size)

def create_model():
    return BasicsTransformerLM(
        10000,
        512,
        1280,
        36,
        20,
        5120,
        10000
    )

def create_optimizer(model : torch.nn.Module):
    return ShardedOptimizer(model.parameters(), AdamW)

def distributed_func(rank, world_size):
    '''
    每个device上执行的函数：
    1. setup device
    2. setup rank
    3. 创建model，和ddp
    4. 创建数据
    5. 创建optimizer
    6. 用shard wrapper optimizer
    7. warmup
    8. train and benchmark
    '''

    torch.cuda.set_device(rank)
    setup(rank, world_size)

    torch.cuda.reset_peak_memory_stats()
    model = create_model().to('cuda')
    ddp = DDP(model)
    torch.cuda.synchronize()
    peak_usage_after_model_init = torch.cuda.max_memory_allocated()

    inputs = torch.randint(low=0, high=10000, size=(2, 512), device='cuda')
    targets = torch.randint(low=0, high=10000, size=(2, 512), device='cuda')

    optimizer:AdamW = create_optimizer(model)

    # warmup
    for _ in range(5):
        optimizer.zero_grad()
        logits = ddp(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        ddp.finish_gradient_sync()
        optimizer.step()
    torch.cuda.reset_peak_memory_stats()

    # benchmark
    it_time_list = []
    peak_mem_before_opt_list = []
    peak_mem_after_opt_list = []
    benchmark_it = 10
    for i in range(benchmark_it):
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        start_it = timeit.default_timer()
        
        optimizer.zero_grad()
        logits = ddp(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        ddp.finish_gradient_sync()

        torch.cuda.synchronize()
        peak_mem_before_opt_list.append(torch.cuda.max_memory_allocated())
        optimizer.step()
        torch.cuda.synchronize()
        peak_mem_after_opt_list.append(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize()
        end_it = timeit.default_timer()

        it_time = end_it - start_it

        it_time_list.append(it_time)

    total_it_time_list = [None] * world_size
    total_init_mem_list = [None] * world_size
    total_mem_before_opt_list = [None] * world_size
    total_mem_after_opt_list = [None] * world_size

    dist.all_gather_object(total_it_time_list, it_time_list)
    dist.all_gather_object(total_init_mem_list, peak_usage_after_model_init)
    dist.all_gather_object(total_mem_before_opt_list, peak_mem_before_opt_list)
    dist.all_gather_object(total_mem_after_opt_list, peak_mem_after_opt_list)

    if rank == 0:
        sum_it = 0
        max_mem_before_opt = 0
        max_mem_after_opt = 0
        for i in range(benchmark_it):
            sum_it += max(total_it_time_list[j][i] for j in range(world_size))
            max_mem_before_opt = max(max(total_mem_before_opt_list[j][i] for j in range(world_size)), max_mem_before_opt)
            max_mem_after_opt = max(max(total_mem_after_opt_list[j][i] for j in range(world_size)),max_mem_after_opt)

        avg_it = sum_it / benchmark_it

        print('avg_it =', avg_it, 's')
        print('init_model_mem =', max(mem for mem in total_init_mem_list ))
        print('max_mem_before_opt =', max_mem_before_opt)
        print('max_mem_after_opt =', max_mem_after_opt)

    dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(fn=distributed_func, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()