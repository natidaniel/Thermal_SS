
"""
Adapted from (cross-entropy versions):
Semantic Segmentation Algorithms Implemented in PyTorch
https://github.com/meetps/pytorch-semseg
"""

import torch
import torch.nn.functional as F



def cross_entropy2d(input, target, class_weights=None, reduction='mean', ignore_indices=[-1]):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()
    ignore_index = ignore_indices[0]
    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    mask = torch.zeros_like(target)
    for index in ignore_indices:
        mask = mask | (target == index)
    target[mask] = ignore_index
    loss = F.cross_entropy(input, target, weight=class_weights, reduction=reduction, ignore_index=ignore_index)
    return loss

