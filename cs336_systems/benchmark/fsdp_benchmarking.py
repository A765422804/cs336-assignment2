import os
import timeit
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.fsdp import FSDP

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
    return AdamW(
        model.parameters()
    )

def distributed_func(rank, world_size):
    '''
    每个device上执行的函数：
    1. setup device
    2. setup rank
    3. 创建model
    4. 用FSDP wrapper
    5. 创建数据
    6. 创建optimizer
    7. warmup
    8. train and benchmark
    '''

    torch.cuda.set_device(rank)
    setup(rank, world_size)

    torch.cuda.reset_peak_memory_stats()
    model = create_model().to('cuda')
    fsdp: FSDP = FSDP(model)
    torch.cuda.synchronize()
    peak_usage_after_model_init = torch.cuda.memory_allocated()

    inputs = torch.randint(low=0, high=10000, size=(2, 512), device='cuda')
    targets = torch.randint(low=0, high=10000, size=(2, 512), device='cuda')

    optimizer:AdamW = create_optimizer(fsdp)

    # warmup
    for _ in range(5):
        optimizer.zero_grad()
        logits = fsdp(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        fsdp.finish_gradient_synchronization()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # benchmark
    it_time_list = []
    peak_mem_list = []
    benchmark_it = 10
    for i in range(benchmark_it):
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        start_it = timeit.default_timer()
        
        optimizer.zero_grad()
        logits = fsdp(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()

        fsdp.finish_gradient_synchronization()

        optimizer.step()

        torch.cuda.synchronize()
        peak_mem_list.append(torch.cuda.max_memory_allocated())
        end_it = timeit.default_timer()

        it_time = end_it - start_it

        it_time_list.append(it_time)

    total_it_time_list = [None] * world_size
    total_peak_mem_list = [None] * world_size
    total_init_mem_list = [None] * world_size

    dist.all_gather_object(total_it_time_list, it_time_list)
    dist.all_gather_object(total_peak_mem_list, peak_mem_list)
    dist.all_gather_object(total_init_mem_list, peak_usage_after_model_init)

    if rank == 0:
        sum_it = 0
        max_mem = 0
        for i in range(benchmark_it):
            sum_it += max(total_it_time_list[j][i] for j in range(world_size))
            max_mem = max(max(total_peak_mem_list[j][i] for j in range(world_size)), max_mem)

        avg_it = sum_it / benchmark_it

        print('avg_it =', avg_it, 's')
        print('init_model_mem =', max(mem for mem in total_init_mem_list) / 1024**3)
        print('max_mem_in_step =', max_mem / 1024**3)

    dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(fn=distributed_func, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()