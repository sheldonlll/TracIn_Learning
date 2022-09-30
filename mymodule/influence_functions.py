import torch


def tracin_get(a, b):
    return sum(torch.dot(at.flattern(), bt.flattern()) for at, bt in zip(a, b))