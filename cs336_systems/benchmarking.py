'''
运行模型训练过程，测试性能指标
'''

import argparse
import torch
import yaml
import timeit
from tqdm import tqdm
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

def create_model(config):
    model = BasicsTransformerLM(
        config['vocab_size'],
        config['context_length'],
        config['d_model'],
        config['num_layers'],
        config['num_heads'],
        config['d_ff'],
        config['rope_theta']
    )

    return model

def create_optimizer(config, model:BasicsTransformerLM):
    optimizer = AdamW(
        model.parameters(),
        config['lr'],
        tuple(config['betas']),
        config['eps'],
        config['weight_decay']
    )

    return optimizer

def run_step(inputs, targets, mode, model, optimizer):
    # forward
    if mode == 'forward':
        with nvtx.range('forward'):
            logits = model(inputs)

    # backward
    elif mode == 'forward_backward':
        optimizer.zero_grad()
        with nvtx.range('forward'):
            logits = model(inputs)
        with nvtx.range('backward'):
            loss = cross_entropy(logits, targets)
            loss.backward()

    # optimizer
    elif mode == 'full_step':
        optimizer.zero_grad()
        with nvtx.range('forward'):
            logits = model(inputs)
        with nvtx.range('backward'):
            loss = cross_entropy(logits, targets)
            loss.backward()
        with nvtx.range('optimizer'):
            optimizer.step()
    else:
        assert(False)

def main():
    # 读取config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        required=True
    )
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 初始化参数
    device = torch.device(config['runtime']['device'])
    batch_size = config['data']['batch_size']
    context_length = config['model']['context_length']
    vocab_size = config['model']['vocab_size']
    warmup_steps = config['benchmark']['warmup_steps']
    measured_steps = config['benchmark']['measured_steps']
    mode = config['benchmark']['mode']

    # 创建模型
    model: BasicsTransformerLM = create_model(config['model']).to(device)

    # 随机 input 和 target 的数据
    inputs = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)
    targets = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)

    # 创建optimizer
    optimizer: AdamW = create_optimizer(config['optimizer'], model)

    time_elapsed_list = []
    warmup_pbar = tqdm(range(warmup_steps))
    measured_pbar = tqdm(range(measured_steps))

    model.train()

    for t in warmup_pbar:
        run_step(inputs, targets, mode, model, optimizer)

        if device.type == 'cuda':
            torch.cuda.synchronize()

    with nvtx.range('measured'):
        for t in measured_pbar:
            # start time
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = timeit.default_timer()

            run_step(inputs, targets, mode, model, optimizer)

            # end time    
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = timeit.default_timer()
            time_elapsed = end_time - start_time
            time_elapsed_list.append(time_elapsed)

            # log
            measured_pbar.set_postfix(time_elapsed=time_elapsed)

    # 计算mean和std
    times = torch.tensor(time_elapsed_list)
    mean = times.mean().item()
    std = times.std().item()

    print('mode=',mode)
    print('mean=',mean)
    print('std=', std)
    print('warmup_stes',warmup_steps)
    print('measured_steps=',measured_steps)


if __name__ == '__main__':
    main()