import torch

# Emulate tensorflow's convert_to_tensorflow function
def convert_to_tensor(obj):
    """
    Takes a nested list of PyTorch tensors with no varying lengths along a dimension.
    Returns one PyTorch tensor created by stacking the nested list elements along the corresponding dimensions
    """
    if type(obj) == torch.Tensor:
        return obj
    elif type(obj) == list:
        return torch.stack([*map(convert_to_tensor, obj)], dim=0)
    else:
        raise TypeError(f'Unconvertible type {type(obj)}')