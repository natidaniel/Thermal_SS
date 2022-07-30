
import functools
import torch.nn as nn

from .losses import (
    cross_entropy2d,
    
)

loss_names = {
    "cross_entropy": cross_entropy2d,
    
}


def get_loss_function(cfg):
    if cfg["training"]["loss"] is None:
        return cross_entropy2d

    else:
        loss_dict = cfg["training"]["loss"]
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        if loss_name not in loss_names:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))
        return functools.partial(loss_names[loss_name], **loss_params)
