# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================

import torch
import torch.nn as nn

import tinycudann as tcnn

from triplane_decoder.interpolation import grid_sample_wrapper
from triplane_decoder.activation import init_density_activation
from triplane_decoder.scene_contraction import contract_world
from triplane_decoder.pif import PIF

class TriplaneDecoder(nn.Module):

    def __init__(self, config, solo=False):
        """
        Args:
            config (dict): Decoder config.
            world2idx (callable): Function that maps world coordinates to voxel indices.
        """
        super(TriplaneDecoder, self).__init__()

        self.config = config

        self.contraction = config.scene_contraction
            
        self.N_z, self.N_h, self.N_w = config.N_z_, config.N_h_, config.N_w_
        self.offset_z, self.offset_h, self.offset_w = config.offset_z, config.offset_h, config.offset_w
        self.scale_z, self.scale_h, self.scale_w = config.scale_z, config.scale_h, config.scale_w

        num_features = config._dim_

        # init planes (xy, yz, xz)
        self.plane_zh = torch.empty((self.N_z, self.N_h, num_features))
        self.plane_hw = torch.empty((self.N_h, self.N_w, num_features))
        self.plane_zw = torch.empty((self.N_z, self.N_w, num_features))  
        
        if solo:
            self.plane_zh = nn.Parameter(self.plane_zh)
            self.plane_hw = nn.Parameter(self.plane_hw)
            self.plane_zw = nn.Parameter(self.plane_zw)
            nn.init.uniform_(self.plane_zh, a=0.1, b=0.5)
            nn.init.uniform_(self.plane_hw, a=0.1, b=0.5)
            nn.init.uniform_(self.plane_zw, a=0.1, b=0.5)

        if config.pif:
            num_features += 128 * 2
            self.bn = nn.BatchNorm1d(128 * 2)

        self.decoder_net = tcnn.Network(
            n_input_dims=num_features,
            n_output_dims=4,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.config.decoder.hidden_dim,
                "n_hidden_layers": self.config.decoder.hidden_layers,
            },
        )

        self.density_activation = init_density_activation(config.decoder.density_activation)    

        self.color_activation = nn.Sigmoid()


    def forward(self, x_world, d=None, pif: PIF=None):
        sigma = torch.zeros_like(x_world[:, 0]) #samples
        c = torch.zeros_like(x_world)           #samples, 3


        # World coordinates -> [-1,1] grid coordinates
        if self.contraction:
            x_grid = contract_world(x_world.clone() * x_world.new_tensor(self.config.scene_contraction_factor)) #samples, 3
        else:
            x_grid = (x_world - torch.tensor([self.offset_z, self.offset_h, self.offset_w], device=x_world.device)) / \
                  (torch.tensor([self.scale_z * self.N_z, self.scale_h * self.N_h, self.scale_w * self.N_w], device=x_world.device) )
            
        if pif is None:
            mask = (x_grid.abs() < 1.0).all(-1) # samples
            x_grid = x_grid[mask]
        else:
            mask = torch.ones(x_world.size(0),device=x_world.device, dtype=bool)


        if len(x_grid) == 0:
            return c, sigma

        # get features from planes
        F_zh = grid_sample_wrapper(self.plane_zh, x_grid[:, :2]) # [batch_size, F] 
        F_hw = grid_sample_wrapper(self.plane_hw, x_grid[:, 1:]) # [batch_size, F]
        F_zw = grid_sample_wrapper(self.plane_zw, x_grid[:, [2,0]]) # [batch_size, F]

        # hadamard product
        features = F_zh * F_hw * F_zw  # [batch_size, F]

        if pif is not None:
            pif_features = pif(x_world.view(-1,3), aggregate=True, num_features_to_keep=2).permute(1,2,0).reshape(x_world.size(0),-1) #samples, #2*img_features
            pif_features = self.bn(pif_features) * (pif_features != 0)
            features = torch.cat((features, pif_features), dim=1)
        
        h =  self.decoder_net(features).float() #valid_samples, (3+1) 


        c[mask], sigma[mask] = h[:, :-1], h[:, -1]

        if self.density_activation is not None:
            sigma[mask] = self.density_activation(sigma[mask])
        if self.color_activation is not None:
            c[mask] = self.color_activation(c[mask])
    
        return c, sigma

    
    def update_planes(self, triplane):
        self.plane_zh = triplane[2].view((self.N_z, self.N_h,-1)).unsqueeze(0).permute(0,3,1,2).squeeze(0).permute(1,2,0)
        self.plane_hw = triplane[0].view((self.N_h, self.N_w ,-1)).unsqueeze(0).permute(0,3,1,2).squeeze(0).permute(1,2,0)
        self.plane_zw = triplane[1].view((self.N_z, self.N_w, -1)).unsqueeze(0).permute(0,3,1,2).squeeze(0).permute(1,2,0)
