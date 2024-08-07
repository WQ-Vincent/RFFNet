import importlib
import torch
from torch import nn
from collections import OrderedDict

def get_model(mode, opt={}):
    if opt == None:
        opt = {}
    print('getting model %s ...' % mode)
    model_ = getattr(importlib.import_module(f'model.{mode}'), mode)
    model = model_(**opt)
    return model

def load_model(model, mode, opt={}, param_key="state_dict"):
    net = get_model(mode, opt)
    checkpoint = torch.load(model)
    net.cuda()
    try:
        net.load_state_dict(checkpoint[param_key])
    except:
        state_dict = checkpoint[param_key]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    return net


if __name__ == "__main__":
    from thop import profile
    from thop import clever_format

    model = get_model()
    # model = get_pretrain()
    input = torch.randn(1, 3, 128, 128)
    ir = torch.randn(1, 1, 128, 128)
    flops, params = profile(model, inputs=(input, ir))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)