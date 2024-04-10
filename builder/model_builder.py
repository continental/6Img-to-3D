# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================

from triplane_encoder import *
from mmseg.models import build_segmentor

def build(model_config):
    model = build_segmentor(model_config)
    model.init_weights()
    return model
