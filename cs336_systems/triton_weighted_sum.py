import torch
import triton
import triton.language as tl
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
    
def weighted_sum(x, weight):
    return WeightedSumFunc.apply(x, weight)

x = torch.randn(4, 512, 128, device='cuda', dtype=torch.float32)
weight = torch.randn(128, device='cuda', dtype=torch.float32)

y_ref = (x * weight).sum(dim=-1)
y_tri = weighted_sum(x, weight)

torch.testing.assert_close(y_tri, y_ref, rtol=1e-4, atol=1e-4)
print('passed')
