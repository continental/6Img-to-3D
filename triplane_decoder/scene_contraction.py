# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================

import torch


    
def contract_world(x):
    """
    Maps world coordinates to a contracted [-1 -> 1] space
    """
    mag = torch.linalg.norm(x, ord=2, dim=-1)
    mask = mag >= 1
    #make it fit in a cube of size 1
    x[mask] = (2 - (1 / mag[mask][..., None])) * (x[mask] / mag[mask][..., None]) 

    x /= 2

    return x      

def uncontract_world(x):
    """
    From a contracted [-1->1] space to world coordinates
    """
    x_shape = x.shape
    x *= 2
    x = x.reshape(-1,3)
    mag = torch.linalg.norm(x, ord=2, dim=-1)
    mask = mag >= 1
    if mask.any():
        x[mask] =  -(x[mask] + (2 * x[mask] * (torch.sqrt(x[mask,0]**2 + x[mask,1]**2 + x[mask,2]**2) + 2)[...,None]) / (x[mask,0]**2 + x[mask,1]**2 + x[mask,2]**2 - 4)[...,None]) / ((x[mask,0]**2 + x[mask,1]**2 + x[mask,2]**2))[...,None]
    x = x.reshape(x_shape)
    return x