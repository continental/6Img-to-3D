# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================

import numpy as np
import torch


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)



def custom_collate_fn(input_data):
    img2stack = torch.from_numpy(np.stack([d[0] for d in input_data]).astype(np.float32)).permute(0,1,4,2,3) if len(input_data[0][0]) > 0 else None
    meta2stack = [d[1] for d in input_data] if input_data[0][1] is not None else None
    dataset_stack = [d[2] for d in input_data] if input_data[0][2] is not None else None
    if len(input_data[0]) > 3:
        label_to_stack = [d[3] for d in input_data]
    return (img2stack, meta2stack, dataset_stack) if len(input_data[0]) < 4 else (img2stack, meta2stack, dataset_stack, label_to_stack)