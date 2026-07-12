import os
import timeit
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.ddp import DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('nccl',rank=rank, world_size=world_size)

def create_model():
    return BasicsTransformerLM(
        10000,
        512,
        2560,
        32,
        32,
        10240,
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
    4. 用DDP wrapper
    5. 创建数据
    6. 创建optimizer
    7. warmup
    8. train and benchmark
    '''

    torch.cuda.set_device(rank)
    setup(rank, world_size)

    model = create_model().to('cuda')
    ddp = DDP(model)

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

    # benchmark
    it_time_list = []
    reduce_time_list = []
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

        torch.cuda.synchronize()
        start_reduce = timeit.default_timer()
        ddp.finish_gradient_sync()
        torch.cuda.synchronize()
        end_reduce = timeit.default_timer()

        optimizer.step()

        torch.cuda.synchronize()
        end_it = timeit.default_timer()

        it_time = end_it - start_it
        reduce_time = end_reduce - start_reduce

        it_time_list.append(it_time)
        reduce_time_list.append(reduce_time)

    total_it_time_list = [None] * world_size
    total_reduce_time_list = [None] * world_size

    dist.all_gather_object(total_it_time_list, it_time_list)
    dist.all_gather_object(total_reduce_time_list, reduce_time_list)

    if rank == 0:
        sum_it = 0
        sum_reduce = 0
        for i in range(benchmark_it):
            sum_it += max(total_it_time_list[j][i] for j in range(world_size))
            sum_reduce += max(total_reduce_time_list[j][i] for j in range(world_size))

        avg_it = sum_it / benchmark_it
        avg_reduce = sum_reduce / benchmark_it
        ratio = avg_reduce / avg_it

        print('avg_it =', avg_it, 's')
        print('avg_reduce =', avg_reduce, 's')
        print('ratio =', ratio * 100, '%')

    dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(fn=distributed_func, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()