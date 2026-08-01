import torch
from cs336_basics.models.optimizers import SGD

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))

for lr in [1e1, 1e2, 1e3]:
    print(f"lr set to {lr}")
    opt = SGD([weights], lr=lr)
    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
