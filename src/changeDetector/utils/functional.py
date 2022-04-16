# Original Author       : Ghasem Abdi, ghasem.abdi@yahoo.com
# File Last Update Date : April 15, 2022

import inspect
import torchinfo
import torch.onnx
import torch.utils

def noop(x): return x

def prepare_dataloader(dataset, batch_size=1, **kwargs): 
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, **kwargs)

def summary(model, input_size, col_names=('input_size', 'output_size', 'num_params'), depth=5, **kwargs):
    torchinfo.summary(model=model, input_size=input_size, col_names=col_names, depth=depth, verbose=1, **kwargs)

def prepare_kwargs(kwargs, func):
    kwargs_, sig = {}, inspect.signature(func)
    for param in sig.parameters.values():
        if param.name in kwargs:
            kwargs_[param.name] = kwargs[param.name]
    return kwargs_

def export_onnx(model, input_size, filename='change detection.onnx', **kwargs):
    torch.onnx.export(model=model, args=(torch.randn(input_size[0]), torch.randn(input_size[1])), f=filename, **kwargs)