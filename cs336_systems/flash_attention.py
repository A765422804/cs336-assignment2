'''
flash attention 的实现
'''

import math
import torch
from torch import Tensor
from jaxtyping import Float
from einops import einsum, rearrange
import triton
import triton.language as tl

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: Float[Tensor, '... n_q d'], K: Float[Tensor, '... n_k d'], V: Float[Tensor, '... n_k d'], is_causal: bool=False)->Float[Tensor, '... n_q d']:
        '''
        flash attention 的pytorch实现
        1. 把 Q K V 按照 tile_size 划分
        2. 遍历所有的Q_tile，初始化当前Q_tile对应的三个属性：
            1. 前 j 个 QK 的score中的 rowmax m
            2. 前 j 个 QK softmax 的分母 l
            3. 前 j 个 QKV 计算得到的分子和 V 的乘积
        3. 对于当前Q_tile，遍历所有的KV，更新属性：
            1. 计算 Q_tile K_tile^T 得到局部 S
            2. 基于局部 S 更新 rowmax m
            3. 基于 m 计算当前 tile 的exp
            4. 使用online softmax更新分母l
            5. 更新分子和 V 的乘积 O
        4. O / l 得到当前tile的计算结果
        5. 计算logsumexp并保存下来方便backward计算梯度
        6. 把当前tile的计算结果写回
        '''

        q_tile_size:int=16
        k_tile_size:int=16
        
        n_q = Q.shape[-2]
        n_k = K.shape[-2]
        d = Q.shape[-1]

        T_q = math.ceil(n_q/ q_tile_size)
        T_k = math.ceil(n_k/ k_tile_size)

        O = torch.empty_like(Q)
        L = torch.empty(size=(Q.shape[:-1]), device=Q.device, dtype=Q.dtype)

        for i in range(T_q):
            q_i = Q[... , i * q_tile_size: min((i + 1) * q_tile_size, n_q), :] # ... q_tile_size d
            m_i = torch.full(size=q_i.shape[:-1], fill_value=-torch.inf, device=Q.device, dtype=Q.dtype) # ... q_tile_size
            o_i = torch.zeros_like(q_i) # ... q_tile_size d
            l_i = torch.zeros(size=q_i.shape[:-1], device=Q.device, dtype=Q.dtype) # ... q_tile_size

            for j in range(T_k):
                k_j = K[..., j * k_tile_size: min((j + 1) * k_tile_size, n_k), :] # ... k_tile_size d
                v_j = V[..., j * k_tile_size: min((j + 1) * k_tile_size, n_k), :] # ... k_tile_size d
                s_i_j = einsum(q_i, k_j, '... q_tile_size d, ... k_tile_size d -> ... q_tile_size k_tile_size') / math.sqrt(d)
                m_i_new = torch.max(m_i, torch.max(s_i_j, dim=-1).values)
                p_i_j = torch.exp(s_i_j - m_i_new[..., :, None]) # ... q_tile_size k_tile_size
                l_i = torch.exp(m_i - m_i_new) * l_i + torch.sum(p_i_j, dim=-1)
                o_i = torch.exp(m_i - m_i_new)[..., :, None] * o_i + p_i_j @ v_j
                m_i = m_i_new

            o_i = o_i / l_i[..., :, None]
            l_i = m_i + torch.log(l_i)

            O[..., i * q_tile_size: min((i + 1) * q_tile_size, n_q), :] = o_i
            L[..., i * q_tile_size: min((i + 1) * q_tile_size, n_q)] = l_i

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_output: Float[Tensor, '... n_q d']):
        Q, K, V, O, L = ctx.saved_tensors

        dQ, dK, dV = flash_attention_backward_torch_helper(Q, K, V, O, grad_output, L, ctx.is_causal)

        return dQ, dK, dV, None

@torch.compile
def flash_attention_backward_torch_helper(Q, K, V, O, dO, L, is_causal):
    '''
    实现backward，按照公式，返回dQ dK dV
    Q: ... n_q d
    K: ... n_k d
    V: ... n_k d
    O: ... n_q d
    dO: ... n_q d
    L: ... n_q 
    is_causal: bool
    '''

    d = Q.shape[-1]

    S = Q.to(torch.float32) @ K.to(torch.float32).transpose(-1, -2) / math.sqrt(d) # ... n_q n_k

    if is_causal:
        n_q = Q.shape[-2]
        n_k = K.shape[-2]
        q_idx = torch.arange(0, n_q, device=Q.device) # n_q
        k_idx = torch.arange(0, n_k, device=K.device) # n_k
        mask = k_idx[None, :] > q_idx[:, None]
        S += torch.where(mask, -1e6, 0.0)

    P = torch.exp(S - L[..., None]) # ... n_q n_k

    dV = P.transpose(-1, -2) @ dO.to(torch.float32) # ... n_k d

    dP = dO.to(torch.float32) @ V.to(torch.float32).transpose(-1, -2) # ...n_q n_k

    D = torch.sum(O * dO, dim=-1) # ... n_q
    dS = P * (dP - D[..., None]) # ... n_q n_k

    dQ = dS @ K.to(torch.float32) / math.sqrt(d) # ... n_q d 

    dK = dS.transpose(-1, -2) @ Q.to(torch.float32) / math.sqrt(d) # ... n_k d

    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)
    
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    '''
    flash attention forward 的 triton 实现。
    一个instance计算单个batch_size和单个Q_TILE_SIZE,D的结果。
    '''

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # 初始化值
    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero') # q_tile d
    m = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # load KV
        K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero') # k_tile d    
        V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero') # k_tile d

        S_tile = tl.dot(Q_tile, tl.trans(K_tile)) * scale # q_tile k_tile
        if is_causal:
            q_start = query_tile_index * Q_TILE_SIZE
            k_start = i * K_TILE_SIZE
            q_idx = q_start + tl.arange(0, Q_TILE_SIZE) # q_tile
            k_idx = k_start + tl.arange(0, K_TILE_SIZE) # k_tile
            casual_mask =  k_idx[None, :] > q_idx[:, None]
            S_tile += tl.where(casual_mask, -1e6, 0.0)
        m_new = tl.maximum(m, tl.max(S_tile, axis=-1)) # q_tile
        P = tl.exp(S_tile - m_new[:, None]) # q_tile k_tile
        l = tl.exp(m - m_new) * l + tl.sum(P, axis=-1)
        acc = tl.dot(P.to(V_tile.dtype), V_tile, acc=tl.exp(m - m_new)[:, None] * acc)
        m = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o = acc / l[:, None] # q_tile d
    l = m + tl.log(l) # q_tile

    o = o.to(O_block_ptr.type.element_ty)
    l = l.to(L_block_ptr.type.element_ty)

    tl.store(O_block_ptr, o, boundary_check=(0, 1))
    tl.store(L_block_ptr, l, boundary_check=(0,))

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: Float[Tensor, '... n_q d'], K: Float[Tensor, '... n_k d'], V: Float[Tensor, '... n_k d'], is_causal: bool=False)->Float[Tensor, '... n_q d']:
        '''
        triton实现，直接调用kernel。
        '''

        q_tile_size:int=16
        k_tile_size:int=16

        q_shape = Q.shape
        k_shape = K.shape

        Q = rearrange(Q, '... n_q d -> (...) n_q d')
        K = rearrange(K, '... n_k d -> (...) n_k d')
        V = rearrange(V, '... n_k d -> (...) n_k d')

        batch_size, n_q, d = Q.shape
        n_k = K.shape[1]

        T_q = math.ceil(n_q / q_tile_size)
        T_k = math.ceil(n_k / k_tile_size)

        O = torch.empty_like(Q)
        L = torch.empty(Q.shape[:-1], device=Q.device, dtype=torch.float32)

        flash_fwd_kernel[(T_q, batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_q, n_k,
            1 / math.sqrt(d),
            d,
            q_tile_size,k_tile_size,
            is_causal
        )

        L = L.reshape(q_shape[:-1])
        O = O.reshape(q_shape)
        Q = Q.reshape(q_shape)
        K = K.reshape(k_shape)
        V = V.reshape(k_shape)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_output: Float[Tensor, '... n_q d']):
        Q, K, V, O, L = ctx.saved_tensors

        dQ, dK, dV = flash_attention_backward_torch_helper(Q, K, V, O, grad_output, L, ctx.is_causal)

        return dQ, dK, dV, None