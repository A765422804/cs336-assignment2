import os
import timeit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo',rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, data_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (data_size // 4,), dtype=torch.float32)

    # warmup
    for _ in range(5):
        dist.all_reduce(data, async_op=False)

    # benchmark
    elapsed_time_list = []
    for _ in range(5):
        dist.barrier()
        start_time = timeit.default_timer()
        dist.all_reduce(data, async_op=False)
        end_time = timeit.default_timer()
        elapsed_time_list.append(end_time - start_time)
    
    # gather
    total_elapsed_time_list = [None] * world_size
    dist.all_gather_object(total_elapsed_time_list, elapsed_time_list)
    if rank == 0:
        print('------world_size =', world_size,' data_size =', data_size,'------')
        rank_mean = [sum(x)/ len(x) for x in total_elapsed_time_list]
        total_mean = sum(sum(x) for x in total_elapsed_time_list) / sum(len(x) for x in total_elapsed_time_list)
        total_max = max(max(x) for x in total_elapsed_time_list)

        print('rank_mean =', rank_mean)
        print('total_mean =',total_mean)
        print('total_max =', total_max)

    dist.destroy_process_group()

if __name__ == '__main__':
    world_size_list = [2, 4, 6]
    data_size_list = [1000000, 10000000, 100000000, 1000000000] # bytes
    for data_size in data_size_list:
        for world_size in world_size_list:
            mp.spawn(fn=distributed_demo, args=(world_size,data_size), nprocs=world_size, join=True)