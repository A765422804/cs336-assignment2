from cs336_systems.flash_attention import FlashAttentionTriton
from cs336_basics.model import scaled_dot_product_attention
import torch
import triton

seq_len_start = 128
seq_len_end = 65536

d_start = 16
d_end = 128

batch_size = 1

precisions = [torch.bfloat16, torch.float32]

def flash_attention(Q, K, V):
    return FlashAttentionTriton.apply(Q,K,V, True) 

def flash_attention_benchmark(Q, K, V, dO):
    out = None
    try:
        # warmup
        out = flash_attention(Q, K, V)
        out.backward(dO)
        Q.grad = None
        K.grad = None
        V.grad = None   

        out = None

        # forward
        def triton_forward_workload(): flash_attention(Q, K, V)
        triton_forward = triton.testing.do_bench(triton_forward_workload)
        print('triton_forward =', triton_forward)

        # backward 
        out = flash_attention(Q, K, V)
        def triton_backward_workload():
            out.backward(dO, retain_graph=True)
        triton_backward = triton.testing.do_bench(triton_backward_workload, grad_to_none=(Q, K, V))
        out = None
        print('triton_backward =', triton_backward)

        # foward and backward
        def triton_fb_workload():
            out = flash_attention(Q, K, V)
            out.backward(dO)
        triton_fb = triton.testing.do_bench(triton_fb_workload, grad_to_none=(Q, K, V))
        print('triton_fb =', triton_fb)

        Q.grad = None
        K.grad = None
        V.grad = None  
    
    except torch.cuda.OutOfMemoryError:
        out = None
        Q.grad = None
        K.grad = None
        V.grad = None  
        torch.cuda.empty_cache()
        print('flash attention torch.cuda.OutOfMemoryError')       

def pytorch_attention_benchmark(Q, K, V, dO, seq_len): 
    out = None
    mask = None
    q_idx = None
    k_idx = None
    try:
        # mask
        q_idx = torch.arange(0, seq_len, device='cuda') # n_q
        k_idx = torch.arange(0, seq_len, device='cuda') # n_k
        mask = k_idx[None, :] <= q_idx[:, None]

        # warmup
        out = scaled_dot_product_attention(Q, K, V, mask)
        out.backward(dO)
        Q.grad = None
        K.grad = None
        V.grad = None

        out = None

        # forward
        def pytorch_forward_workload(): scaled_dot_product_attention(Q, K, V, mask)
        pytorch_forward = triton.testing.do_bench(pytorch_forward_workload)
        print('pytorch_forward =', pytorch_forward)

        # backward 
        out = scaled_dot_product_attention(Q, K, V, mask)
        def pytorch_backward_workload():
            out.backward(dO, retain_graph=True)
        pytorch_backward = triton.testing.do_bench(pytorch_backward_workload, grad_to_none=(Q, K, V))
        out = None
        print('pytorch_backward =',pytorch_backward)

        # foward and backward
        def pytorch_fb_workload():
            out = scaled_dot_product_attention(Q, K, V, mask)
            out.backward(dO)
        pytorch_fb = triton.testing.do_bench(pytorch_fb_workload, grad_to_none=(Q, K, V))
        print('pytorch_fb =', pytorch_fb)

        Q.grad = None
        K.grad = None
        V.grad = None  
    
    except torch.cuda.OutOfMemoryError:
        out = None
        mask = None
        q_idx = None
        k_idx = None
        Q.grad = None
        K.grad = None
        V.grad = None  
        torch.cuda.empty_cache()
        print('pytorch attention torch.cuda.OutOfMemoryError')   

seq_len = seq_len_start
while seq_len <= seq_len_end:
    d = d_start
    while d <= d_end:
        for precision in precisions:
            print("------ seq_len =", seq_len,' d =', d, ' precision =', precision, ' ------')

            Q = None
            K = None
            V = None
            dO = None
            try:
                '''
                测试两种attention计算：
                Q: ... n_q d
                K/V: ... n_k d
                '''
                Q = torch.randn(batch_size, seq_len, d, device='cuda', dtype=precision, requires_grad=True)
                K = torch.randn(batch_size, seq_len, d, device='cuda', dtype=precision, requires_grad=True)
                V = torch.randn(batch_size, seq_len, d, device='cuda', dtype=precision, requires_grad=True)
                dO = torch.randn(batch_size, seq_len, d, device='cuda', dtype=precision)

                flash_attention_benchmark(Q, K, V, dO)
                pytorch_attention_benchmark(Q, K, V, dO, seq_len)
            except torch.cuda.OutOfMemoryError:
                Q = None
                K = None
                V = None
                dO = None
                torch.cuda.empty_cache()
                print('QKV create torch.cuda.OutOfMemoryError')
        d *=2
    seq_len*=2