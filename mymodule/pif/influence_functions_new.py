import torch
def get_gradient(grads, model):
    """
    pick the gradients by name.
    """
    return [grad for grad, (n, p) in zip(grads, model.named_parameters())]
            # if 'layer.10.' in n or 'layer.11.' in n
            # or 'classifier.' in n or 'pooler.' in n

def tracin_get(a, b):
    """ dot product between two lists"""
    return sum([torch.dot(at.flatten(), bt.flatten()) for at, bt in zip(a, b)])
    # breakpoint()
