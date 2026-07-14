import torch
from torch import nn, Tensor, Size
import torch.distributed as dist
import math
from jaxtyping import Float, Int
from einops import einsum
from cs336_basics.model import Linear, Embedding

class FSDPLinearFunction(torch.autograd.Function):
    '''
    FSDP的特化前向和反向。
    '''
    @staticmethod
    def forward(ctx, x: Float[Tensor, '... in_dim'], local_weight: Float[Tensor, 'numel_per_rank'], gather_weight: Float[Tensor, 'out_dim in_dim'], wrapper: 'FSDPLinear')->Float[Tensor,'... out_dim']:
        '''
        前向函数，实现前向计算和梯度相关变量存储
        '''
        out = x.to(gather_weight.dtype) @ gather_weight.T
        ctx.save_for_backward(x, local_weight)
        ctx.wrapper = wrapper
        return  out

    @staticmethod
    def backward(ctx, out_grad:Float[Tensor, '... out_dim'])->tuple[Float[Tensor, '... in_dim'], Float[Tensor, 'num_per_rank'], None, None]:
        '''
        backward，实现反向计算和local_weight的梯度计算
        - 通过gather_weight计算梯度
        - flatten and padding
        - reduce-scatter得到local_dW
        - 除以world_size
        - 清理临时的全量参数
        '''
        x, local_weight = ctx.saved_tensors
        gather_weight = ctx.wrapper.gather_weight

        dx: Tensor = einsum(out_grad, gather_weight, '... d_out, d_out d_in -> ... d_in')
        dW: Tensor = einsum(out_grad, x.to(out_grad.dtype), ' ... d_out, ... d_in -> d_out d_in ')

        flatten_dW = dW.flatten()
        pad_amount = local_weight.numel() * dist.get_world_size() - flatten_dW.numel()
        flatten_dW = nn.functional.pad(flatten_dW, pad=[0, pad_amount])

        local_dW = torch.empty(size=local_weight.size(), device=local_weight.device, dtype=flatten_dW.dtype)
        dist.reduce_scatter_tensor(
            local_dW,
            flatten_dW,
            dist.ReduceOp.SUM,
            async_op=False
        )

        local_dW = local_dW.to(local_weight.dtype) / dist.get_world_size()

        ctx.wrapper.gather_weight = None
        ctx.wrapper.flatten_weight = None

        return dx.to(x.dtype), local_dW, None, None
    
class FSDPEmbeddingFunction(torch.autograd.Function):
    '''
    FSDP的特化前向和反向。
    '''
    @staticmethod
    def forward(ctx, x: Int[Tensor, '...'], local_weight: Float[Tensor, 'numel_per_rank'], gather_weight: Float[Tensor, 'num_embedding embedding_dim'], wrapper: 'FSDPEmbedding')->Float[Tensor,'... embedding_dim']:
        '''
        前向函数，实现前向计算和梯度相关变量存储
        '''
        out = gather_weight[x]
        ctx.save_for_backward(x, local_weight)
        ctx.wrapper = wrapper
        return  out

    @staticmethod
    def backward(ctx, out_grad:Float[Tensor, '... embedding_dim'])->tuple[None, Float[Tensor, 'num_per_rank'], None, None]:
        '''
        backward，实现反向计算和local_weight的梯度计算
        - 通过gather_weight计算梯度
        - flatten and padding
        - reduce-scatter得到local_dW
        - 除以world_size
        - 清理临时的全量参数
        '''
        x, local_weight = ctx.saved_tensors
        gather_weight = ctx.wrapper.gather_weight

        dW = torch.zeros_like(gather_weight).index_add_(0, x.reshape(-1), out_grad.reshape(-1, gather_weight.shape[1]))

        flatten_dW = dW.flatten()
        pad_amount = local_weight.numel() * dist.get_world_size() - flatten_dW.numel()
        flatten_dW = nn.functional.pad(flatten_dW, pad=[0, pad_amount])

        local_dW = torch.empty(size=local_weight.size(), device=local_weight.device, dtype=flatten_dW.dtype)
        dist.reduce_scatter_tensor(
            local_dW,
            flatten_dW,
            dist.ReduceOp.SUM,
            async_op=False
        )

        local_dW = local_dW.to(local_weight.dtype) / dist.get_world_size()

        ctx.wrapper.gather_weight = None
        ctx.wrapper.flatten_weight = None

        return None, local_dW, None, None

class FSDPLinear(Linear):
    '''
    为了实现FSDP的Linear的wrapper类，保存自己的局部sharded参数。
    每次forward时使用动态gather的weight来进行计算。
    '''

    def __init__(self, local_weight: Float[Tensor, 'numel_per_rank'], source_shape: Size, idx:int ,pad_amount: int,compute_dtype: torch.dtype | None=None):
        nn.Module.__init__(self)

        self.gather_weight = None # out_feature in_feature
        self.flatten_weight = None # numel_per_rank * world_size
        self.local_weight = nn.Parameter(local_weight) # numel_per_rank
        self.compute_local_weight = None # numel_per_rank
        self.handle = None
        self.source_shape = source_shape
        self.pad_amount = pad_amount
        self.compute_dtype = compute_dtype
        self.idx = idx

    def forward(self, X):
        out = FSDPLinearFunction.apply(X, self.local_weight, self.gather_weight, self)
        self.gather_weight = None
        self.flatten_weight = None

        return out
    
    def start_all_gather(self):
        '''
        开始调用all_gather，汇总接下来forward需要的gather_weight
        '''
        with torch.no_grad():
            compute_dtype = self.compute_dtype if self.compute_dtype is not None else self.local_weight.dtype
            self.flatten_weight = torch.empty((self.local_weight.shape[0] * dist.get_world_size()), device=self.local_weight.device, dtype=compute_dtype)
            self.compute_local_weight = self.local_weight.to(compute_dtype)
            self.handle = dist.all_gather_into_tensor(self.flatten_weight, self.compute_local_weight, async_op=True)

    def end_all_gather(self):
        '''
        在gather完毕后，forward之前的代码逻辑，需要基于gather结果得到foward的权重gather_weight
        '''
        with torch.no_grad():
            self.handle.wait()
            self.compute_local_weight = None
            self.handle = None

            valid_flatten_weight = self.flatten_weight[:self.flatten_weight.numel() - self.pad_amount]
            self.gather_weight = valid_flatten_weight.reshape(self.source_shape)        
    
    def pre_forward(self, module, inputs):
        '''
        pre_forward的hook函数
        '''
        self.end_all_gather()
    
    def pre_backward(self, module, grad_out):
        '''
        pre_backward的hook函数
        '''        
        self.end_all_gather()
    
class FSDPEmbedding(Embedding):
    '''
    同上，实现embedding的wrapper类，保存自己的局部参数。
    forward时使用动态gather的weight来进行计算。
    '''

    def __init__(self, local_weight: Float[Tensor, 'numel_per_rank'], source_shape: Size, pad_amount: int, idx:int, compute_dtype: torch.dtype | None=None):
        nn.Module.__init__(self)

        self.gather_weight = None # num_embeddings embedding_dim
        self.flatten_weight = None # numel_per_rank * world_size
        self.local_weight = nn.Parameter(local_weight) # numel_per_rank
        self.compute_local_weight = None # numel_per_rank
        self.handle = None
        self.source_shape = source_shape
        self.pad_amount = pad_amount
        self.compute_dtype = compute_dtype
        self.idx = idx

    def forward(self, x):
        out = FSDPEmbeddingFunction.apply(x, self.local_weight, self.gather_weight, self)
        self.gather_weight = None
        self.flatten_weight = None
        return out
    
    def start_all_gather(self):
        '''
        开始调用all_gather，汇总接下来forward需要的gather_weight
        '''
        with torch.no_grad():
            compute_dtype = self.compute_dtype if self.compute_dtype is not None else self.local_weight.dtype
            self.flatten_weight = torch.empty((self.local_weight.shape[0] * dist.get_world_size()), device=self.local_weight.device, dtype=compute_dtype)
            self.compute_local_weight = self.local_weight.to(compute_dtype)
            self.handle = dist.all_gather_into_tensor(self.flatten_weight, self.compute_local_weight, async_op=True)

    def end_all_gather(self):
        '''
        在gather完毕后，forward之前的代码逻辑，需要基于gather结果得到foward的权重gather_weight
        '''
        with torch.no_grad():
            self.handle.wait()
            self.compute_local_weight = None
            self.handle = None

            valid_flatten_weight = self.flatten_weight[:self.flatten_weight.numel() - self.pad_amount]
            self.gather_weight = valid_flatten_weight.reshape(self.source_shape)        
    
    def pre_forward(self, module, inputs):
        '''
        pre_forward的hook函数
        '''
        self.end_all_gather()
    
    def pre_backward(self, module, grad_out):
        '''
        pre_backward的hook函数
        '''        
        self.end_all_gather()

class FSDP(nn.Module):
    '''
    实现完整的device之间的weight分片，从而达到梯度分片和optimizer state分片的目的
    '''
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None=None):
        '''
        FSDP的初始化函数：
        - 广播 rank 0 的参数
        - 递归遍历module的所有named_modules()，然后识别Linear和embedding层
        - 创建对应的wrapper模块，同时为所有wrapper模块增加pre forward和post forward的回调
            - pre forward回调在wrapper内部，执行all-gather后的数据到forward可用的数据的转换
            - post forward回调在FSDP内部，执行第i+2的wrapper的all-gather的提前overlap调用
        - 记录模块路径和对应wrapper，遍历结束后替换掉对应的原始模块

        对于第i个wrapper的流程：
            第i - 2个wrapper的post forward()
            第i个wrapper的start_all_gather() 开始参数合并
            第i个wrapper的pre_forward() 把合并的参数变成可以运算的shape
            第i个wrapper的forward() 调用自行实现的forward来和backward对应，并在调用后清空全量weight
            第i个wrapper的post_forward()，同时激活第i + 2个wrapper的start_all_gather()
        backward也是同理，需要手动提前启动gather。
        - 对于未分片的参数，还是走传统的ddp路径，即增加一个梯度处理完毕后all-reduce的hook，然后在finish_gradient_sync里面做后续处理
        '''
        super().__init__()

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        self.module = module

        for parameter in module.parameters():
            with torch.no_grad():
                dist.broadcast(parameter, 0)

        self.wrapper_infos = []
        self.wrappers = []
        idx = 0
        for named_module in module.named_modules():
            name, submodule = named_module
            if isinstance(submodule, Linear):
                weight = submodule.weight
                weight_shape = weight.shape

                flatten_weight = weight.flatten()
                flatten_numel = flatten_weight.shape[0]

                # 右侧补0，保证weight是world_size整数倍，每个rank分下来的一样多
                numel_per_rank = math.ceil(flatten_numel / world_size)
                pad_amount = numel_per_rank * world_size - flatten_numel
                flatten_weight = nn.functional.pad(flatten_weight, pad=[0, pad_amount]) # numel_per_rank * world_size

                cur_rank_weight = flatten_weight[rank * numel_per_rank: (rank + 1) * numel_per_rank].detach().clone() # numel_per_rank

                # 创建对应的FSDP的wrapper类
                linear_wrapper = FSDPLinear(
                    local_weight=cur_rank_weight,
                    source_shape = weight_shape,
                    pad_amount=pad_amount,
                    compute_dtype=compute_dtype,
                    idx=idx
                )
                idx += 1

                linear_wrapper.register_forward_pre_hook(linear_wrapper.pre_forward)
                linear_wrapper.register_forward_hook(self.post_forward)

                linear_wrapper.register_full_backward_pre_hook(linear_wrapper.pre_backward)
                linear_wrapper.register_full_backward_hook(self.post_backward)

                self.wrapper_infos.append((name, linear_wrapper))
                self.wrappers.append(linear_wrapper)

            elif isinstance(submodule, Embedding):
                weight = submodule.weight
                weight_shape = weight.shape

                flatten_weight = weight.flatten()
                flatten_numel = flatten_weight.shape[0]

                # 右侧补0，保证weight是world_size整数倍，每个rank分下来的一样多
                numel_per_rank = math.ceil(flatten_numel / world_size)
                pad_amount = numel_per_rank * world_size - flatten_numel
                flatten_weight = nn.functional.pad(flatten_weight, pad=[0, pad_amount]) # numel_per_rank * world_size

                cur_rank_weight = flatten_weight[rank * numel_per_rank: (rank + 1) * numel_per_rank].detach().clone() # numel_per_rank

                # 创建对应的FSDP的wrapper类
                embedding_wrapper = FSDPEmbedding(
                    local_weight=cur_rank_weight,
                    source_shape = weight_shape,
                    pad_amount=pad_amount,
                    compute_dtype=compute_dtype,
                    idx=idx
                )
                idx += 1

                embedding_wrapper.register_forward_pre_hook(embedding_wrapper.pre_forward)
                embedding_wrapper.register_forward_hook(self.post_forward)

                embedding_wrapper.register_full_backward_pre_hook(embedding_wrapper.pre_backward)
                embedding_wrapper.register_full_backward_hook(self.post_backward)

                self.wrapper_infos.append((name, embedding_wrapper))
                self.wrappers.append(embedding_wrapper)

        self.wrapper_param_id_set = set()
        for wrapper_info in self.wrapper_infos:
            module.set_submodule(*wrapper_info)
            self.wrapper_param_id_set.add(id(wrapper_info[1].local_weight))

        # 注册整个backward启动前的回调，提前开始最后两个wrapper的gather
        self.register_full_backward_pre_hook(self.pre_full_backward)

        # 对于未分片参数，注册梯度就绪hook
        self.handle_param_list = []
        for param in module.parameters():
            if id(param) not in self.wrapper_param_id_set:
                param.register_post_accumulate_grad_hook(self.post_param_grad_backward)

    def post_param_grad_backward(self, param: nn.Parameter):
        '''
        hook 函数，在当前参数梯度backward结束时调用
        '''
        with torch.no_grad():
            handle = dist.all_reduce(param.grad, async_op=True)
            self.handle_param_list.append((handle, param))

    def forward(self, *inputs, **kwargs):
        if len(self.wrappers) > 0:
            self.wrappers[0].start_all_gather()
        if len(self.wrappers) > 1:
            self.wrappers[1].start_all_gather()

        return self.module(*inputs, **kwargs)
    
    def post_forward(self, module: FSDPLinear | FSDPEmbedding, inputs, output):
        '''
        执行第i + 2个wrapper的start_all_gather的调用
        '''

        next_idx = module.idx + 2
        if next_idx < len(self.wrappers):
            self.wrappers[next_idx].start_all_gather()

    def post_backward(self, module: FSDPLinear | FSDPEmbedding, grad_input, grad_output):
        '''
        执行第i - 2个wrapper的start_all_gather的调用
        '''

        next_idx = module.idx - 2
        if next_idx >= 0:
            self.wrappers[next_idx].start_all_gather()

    def pre_full_backward(self, module, grad_output):
        '''
        整个backward开始前的回调，启动最末尾两个wrapper的gather
        '''
        if len(self.wrappers) > 0:
            self.wrappers[-1].start_all_gather()
        if len(self.wrappers) > 1:
            self.wrappers[-2].start_all_gather()

    def finish_gradient_synchronization(self):
        '''
        backward 之后 step 之前，把所有的device的没有分片的参数的梯度平均
        '''
        world_size = dist.get_world_size()
        
        with torch.no_grad():
            for handle, parameter in self.handle_param_list:
                handle.wait()
                parameter.grad /= world_size 
        
        # 清空handle list
        self.handle_param_list = []
