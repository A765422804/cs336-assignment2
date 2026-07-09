import torch
import timeit
from cs336_basics.model import scaled_dot_product_attention

batch_size = 8
d_model_list = [16, 32, 64, 128]
sequence_length_list = [256, 1024, 4096, 8192, 16384]
device = torch.device('cuda')
warmup_steps = 20

compiled_attn = torch.compile(scaled_dot_product_attention)

for d_model in d_model_list:
    for sequence_length in sequence_length_list:
        try:
            Q = torch.rand(size=(batch_size, sequence_length, d_model), device=device, requires_grad=True)
            K = torch.rand(size=(batch_size, sequence_length, d_model), device=device, requires_grad=True)
            V = torch.rand(size=(batch_size, sequence_length, d_model), device=device, requires_grad=True)

            total_forward_time = 0
            total_backward_time = 0
            total_memory_usage = 0
            total_peak_memory_usage = 0

            for i in range(warmup_steps):
                output = compiled_attn(Q, K, V)
                loss = output.sum()
                loss.backward()

                Q.grad = None
                K.grad = None
                V.grad = None
                del output, loss

            for i in range(100):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                
                # forward
                forward_start_time = timeit.default_timer()
                output = compiled_attn(Q, K, V)
                loss = output.sum()
                torch.cuda.synchronize()
                forward_end_time = timeit.default_timer()
                total_forward_time += forward_end_time - forward_start_time

                memory_before_backward = torch.cuda.memory_allocated()
                total_memory_usage += memory_before_backward / 1024 ** 2

                # backward
                backward_start_time = timeit.default_timer()
                loss.backward()
                torch.cuda.synchronize()
                backward_end_time = timeit.default_timer()
                total_backward_time += backward_end_time - backward_start_time

                peak_memory = torch.cuda.max_memory_allocated()
                total_peak_memory_usage += peak_memory / 1024 ** 2
                Q.grad = None
                K.grad = None
                V.grad = None
                del output, loss

            print('d_model=',d_model," seq_len=",sequence_length)
            print('avg_forward_time=', total_forward_time / 100, ' avg_backward_time=', total_backward_time / 100, 'total_memory_usage=', total_memory_usage / 100,' peak_memory_usage=', total_peak_memory_usage / 100)
            print('--------------------------------------------------------------')
        
        except torch.cuda.OutOfMemoryError:
            print(f"d_model={d_model}, seq_len={sequence_length}: OOM")
            print('--------------------------------------------------------------')
            Q = K = V = output = loss = None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            continue