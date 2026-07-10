import torch
import triton
import triton.language as tl
import timeit
from einops import rearrange

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,
    output_ptr,
    x_stride_row, x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr
):
    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0)
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        (D,),
        (weight_stride_dim,),
        (0,),
        (D_TILE_SIZE,),
        (0,)
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr, 
        (NUM_ROWS,),
        (output_stride_row,),
        (row_tile_idx * ROWS_TILE_SIZE,),
        (ROWS_TILE_SIZE,),
        (0,)
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option='zero')

        output += tl.sum(row * weight[None, :], axis=1)

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    tl.store(output_block_ptr, output, boundary_check=(0,))

@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,
    grad_output_ptr,
    grad_x_ptr, partial_grad_weight_ptr,
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    grad_output_block_ptr = tl.make_block_ptr(
        base=grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,)
    )

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0)
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0, ),
        block_shape=(D_TILE_SIZE,),
        order=(0,)
    )

    grad_x_block_ptr = tl.make_block_ptr(
        base=grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0)
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        base=partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0)
    )

    grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option='zero') # ROWS_TILE_SIZE

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # 计算对于x_i_j的梯度 grad_i * w_j
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option='zero') # D_TILE_SIZE
        grad_x_row = grad_output[:, None] * weight[None, :] # ROWS_TILE_SIZE D_TILE_SIZE
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0,1))

        # 计算对于部分rows的weight的梯度 sum_i grad_i * x_i_j
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero') # ROWS_TILE_SIZE, D_TILE_SIZE
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE, ))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
    

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        D, output_dims = x.shape[-1], x.shape[:-1]

        input_shape = x.shape
        x = rearrange(x, '... d -> (...) d')

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D
        assert x.is_cuda and weight.is_cuda
        assert x.is_contiguous()

        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape

        n_rows = x.shape[0]
        y = torch.empty((n_rows,), device=x.device)

        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            NUM_ROWS=n_rows,D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE
        )

        return y.view(input_shape[:-1])
    
    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows,D = x.shape
        grad_out = grad_out.view((n_rows,)).contiguous()

        partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight, grad_out, grad_x, partial_grad_weight, 
            x.stride(0), x.stride(1), 
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0),grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            n_rows, D,
            ROWS_TILE_SIZE, D_TILE_SIZE
        )
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x.view(ctx.input_shape), grad_weight
    
def weighted_sum(x, weight):
    return WeightedSumFunc.apply(x, weight)

x_base = torch.randn(4, 128, 128, device='cuda', dtype=torch.float32)
weight_base = torch.randn(128, device='cuda', dtype=torch.float32)

x_ref = x_base.clone().requires_grad_(True)
weight_ref = weight_base.clone().requires_grad_(True)

x_tri = x_base.clone().requires_grad_(True)
weight_tri = weight_base.clone().requires_grad_(True)

start = timeit.default_timer()

y_ref = (x_ref * weight_ref).sum(dim=-1)
y_ref.sum().backward()

end = timeit.default_timer()
print('ref_time',end-start)

start = timeit.default_timer()

y_tri = weighted_sum(x_tri, weight_tri)
y_tri.sum().backward()

end = timeit.default_timer()
print('tri_time',end-start)

torch.testing.assert_close(y_tri, y_ref, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(x_ref.grad, x_tri.grad, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(weight_ref.grad, weight_tri.grad, rtol=1e-4, atol=1e-4)
print('passed')

