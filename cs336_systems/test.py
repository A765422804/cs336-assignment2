import torch
from torch import nn
from cs336_basics.nn_utils import cross_entropy

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        print('first feed-forward layer', x.dtype)
        x = self.relu(x)
        print('relu', x.dtype)
        x = self.ln(x)
        print('layer norm', x.dtype)
        x = self.fc2(x)

        return x

def main():
    device = torch.device('cuda')

    model = ToyModel(10, 10).to(
        device=device,
        dtype=torch.float32
    )
    x = torch.rand(size=(10, 10), device=device, dtype=torch.float32)
    target = torch.randint(low=0,high=10,size=(10,), device=device)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        y = model(x)
        print('model parameter:', next(model.parameters()).dtype)
        print('logits:', y.dtype)

        loss = cross_entropy(y, target)
        print('loss:', loss.dtype)

        loss.backward()
        print('gradient', next(model.parameters()).grad.dtype)

if __name__ == '__main__':
    main()