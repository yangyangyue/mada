import torch
from torch import nn
from torchviz import make_dot


def plot_model(name: str, model: nn.Module, shape: tuple, device: torch.device):
    aggs = torch.randn(shape, device=device)
    apps = torch.randn(shape, device=device)
    status = torch.randn(shape, device=device)
    loss = model(aggs, apps, status)
    g = make_dot(loss)
    g.render(name, view=False)