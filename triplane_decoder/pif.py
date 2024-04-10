# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Théo Gieruc and Marius Kästingschäfer
# ==============================================================================

from typing import Optional, Union
import torch
from torch import device
from torch.nn import functional as F

class PIF(torch.nn.Module):

    def __init__(self, focal_length, principal_point, image_size, c2w):
        """
        focal_length: (2)
        principal_point: (2)
        image_size: (2)
        c2w: (N,4,4)
        """
        super().__init__()        

        self.width = image_size[0]
        self.height = image_size[1]

        K = torch.zeros((4,4), device=focal_length.device) # (C,4,4)
        K[0,0] = focal_length[0]
        K[1,1] = focal_length[1]
        K[2,2] = 1
        K[0,2] = principal_point[0]
        K[1,2] = principal_point[1]
        K[3,3] = 1

        w2c = torch.linalg.inv(c2w) # (C,4,4)

        self.proj_mat = K[None,:3,:].matmul((w2c[:,:,:])).squeeze(0) # (C,3,4)

        self.imgs = None

    def update_imgs(self, imgs: torch.Tensor) -> None:
        """
        Update the images used for the pif
        imgs: (N,C,H,W)
        """
        self.imgs = imgs.squeeze(0)


    def get_uvs(self, UVW: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Project points to the image plane and return the corresponding UV coordinates.
        UVW: (N,C,3)

        Returns:
            uvw: (N,C,3)
            valid: (N,C)
        """

        # Unsqueeze once to avoid unnecessary memory allocations
        UVW_extended = torch.cat((UVW, torch.ones_like(UVW[...,0:1])), dim=-1)  # (N,4)

        # UVW = self.proj_mat[None,:].matmul(UVW_extended[:,None,:,None]).squeeze(-1).permute(1,0,2) # (N,C,3)
        UVW = torch.einsum('ijk,lk->ilj', self.proj_mat, UVW_extended) # (N,C,3)

        # Transform to normalized coordinates
        UVW[...,:2] /= UVW[...,2:]  # (N,C,3,1)

        # Remap UV coordinates to range [0, width]
        UVW[..., 0] *= -1
        UVW[..., 0] += self.height

        # Determine valid pixel coordinates
        valid = ((UVW[..., 0] >= 0) & (UVW[..., 0] < self.height) &
                        (UVW[..., 1] >= 0) & (UVW[..., 1] < self.width) &
                        ( UVW[...,2] < 0))

        return UVW, valid

  

    def forward(self, points_xyz: torch.Tensor, aggregate=False, num_features_to_keep=2) -> torch.Tensor:
        """
        points_xyz: (N,3)

        return: 
            features: (N,C,3)
        """

        uvs, valid = self.get_uvs(points_xyz)
        
        uvs[...,:2] /= (torch.tensor([self.height,self.width], device=self.imgs.device).view(1,2) / 2) 
        uvs[...,:2] -= 1
        
        features = F.grid_sample(self.imgs, uvs[:,None,:,:2], mode='bilinear', padding_mode='border', align_corners=False).squeeze(2).permute(0,2,1)
        features *= valid[...,None].float()

        if aggregate:
            return self.aggregate(features, valid, num_features_to_keep=num_features_to_keep)

        return features
    
    def cuda(self, device: Optional[Union[int, device]] = None) -> "PIF":
        self.proj_mat = self.proj_mat.cuda(device)
        if self.imgs is not None:
            self.imgs = self.imgs.cuda(device)
        return super().cuda(device)

    @staticmethod
    def aggregate(features, valid, num_features_to_keep=2):
        return features.gather(0, valid.int().argsort(0, descending=True)[:num_features_to_keep,:].unsqueeze(-1).expand(-1,-1,features.size(-1)))

  
def batch_project(UVW: torch.tensor, proj_mat, img_hw):
        """
        Project points to the image plane and return the corresponding UV coordinates.
        UVW: (B, N,3)
        proj_mat: (B, num_cam, 3,4)
        img_hw: (B, num_cam, 2)

        Returns:
            uvw: (B, num_cam, N,3)
            valid: (B, num_cam, N)
        """

        # Unsqueeze once to avoid unnecessary memory allocations
        UVW_extended = torch.cat((UVW, torch.ones_like(UVW[...,0:1])), dim=-1)  # (B, N, 4)

        # UVW = self.proj_mat[None,:].matmul(UVW_extended[:,None,:,None]).squeeze(-1).permute(1,0,2) # (N,C,3)
        UVW = torch.einsum('bijk,blk->bilj', proj_mat, UVW_extended) # (B, num_cam, N, 3)

        # Transform to normalized coordinates
        UVW[...,:2] /= UVW[...,2:]  # (B,num_cam,N,3)

        # Remap UV coordinates to range [0,1]
        UVW[..., 1] *= -1
        UVW[..., 1] += img_hw[:,:,None,1]

        UVW[...,:2] /= img_hw[:,:,None]

        # Determine valid pixel coordinates
        valid = ((UVW[...,:2] >=0.0) & (UVW[...,:2] < 1.0)).all(-1) & (UVW[...,2]<1e-5)

        return UVW, valid 