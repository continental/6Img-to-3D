"""
This code is based on TPVFormer https://github.com/wzzheng/TPVFormer


Please find the Apache 2 license conditions here:

https://github.com/wzzheng/TPVFormer/blob/a1cf223ae4b79f56a2b046016c35a8fb3a0b6284/LICENSE
"""



from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
from triplane_decoder.scene_contraction import contract_world
from triplane_decoder.intrinsics import Intrinsics
import triplane_decoder.ray_utils as ray_utils

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module(force=True)
class TPVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross attention.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    def __init__(self, *args, tpv_h, tpv_w, tpv_z, 
                 num_points_in_pillar=[4, 32, 32], 
                 num_points_in_pillar_cross_view=[32, 32, 32],
                 return_intermediate=False, scene_contraction=False, scene_contraction_factor=None, scale=1, offset=0, intrin_factor=0.125, **kwargs):

        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.num_points_in_pillar = num_points_in_pillar
        assert num_points_in_pillar[1] == num_points_in_pillar[2] and num_points_in_pillar[1] % num_points_in_pillar[0] == 0
        self.fp16_enabled = False

        cross_view_ref_points = self.get_cross_view_ref_points(tpv_h, tpv_w, tpv_z, num_points_in_pillar_cross_view)
        self.register_buffer('cross_view_ref_points', cross_view_ref_points)
        self.num_points_cross_view = num_points_in_pillar_cross_view

        self.scene_contraction = scene_contraction
        self.scene_contraction_factor = scene_contraction_factor
        self.intrin_factor = intrin_factor
        self.scale = scale
        self.offset = offset
        self.grid = None
        self.mask = None
        self.ref_3d_hw_uvs = self.ref_3d_hw_mask = None
        self.ref_3d_zh_uvs = self.ref_3d_zh_mask = None
        self.ref_3d_wz_uvs = self.ref_3d_wz_mask = None


    @staticmethod
    def get_cross_view_ref_points(tpv_h, tpv_w, tpv_z, num_points_in_pillar):
        # ref points generating target: (#query)hw+zh+wz, (#level)3, #p, 2
        # generate points for hw and level 1
        h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h
        w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w
        h_ranges = h_ranges.unsqueeze(-1).expand(-1, tpv_w).flatten()
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_h, -1).flatten()
        hw_hw = torch.stack([w_ranges, h_ranges], dim=-1) # hw, 2
        hw_hw = hw_hw.unsqueeze(1).expand(-1, num_points_in_pillar[2], -1) # hw, #p, 2
        # generate points for hw and level 2
        z_ranges = torch.linspace(0.5, tpv_z-0.5, num_points_in_pillar[2]) / tpv_z # #p
        z_ranges = z_ranges.unsqueeze(0).expand(tpv_h*tpv_w, -1) # hw, #p
        h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h
        h_ranges = h_ranges.reshape(-1, 1, 1).expand(-1, tpv_w, num_points_in_pillar[2]).flatten(0, 1)
        hw_zh = torch.stack([h_ranges, z_ranges], dim=-1) # hw, #p, 2
        # generate points for hw and level 3
        z_ranges = torch.linspace(0.5, tpv_z-0.5, num_points_in_pillar[2]) / tpv_z # #p
        z_ranges = z_ranges.unsqueeze(0).expand(tpv_h*tpv_w, -1) # hw, #p
        w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w
        w_ranges = w_ranges.reshape(1, -1, 1).expand(tpv_h, -1, num_points_in_pillar[2]).flatten(0, 1)
        hw_wz = torch.stack([z_ranges, w_ranges], dim=-1) # hw, #p, 2
        
        # generate points for zh and level 1
        w_ranges = torch.linspace(0.5, tpv_w-0.5, num_points_in_pillar[1]) / tpv_w
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_z*tpv_h, -1)
        h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h
        h_ranges = h_ranges.reshape(1, -1, 1).expand(tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        zh_hw = torch.stack([w_ranges, h_ranges], dim=-1)
        # generate points for zh and level 2
        z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, tpv_h, num_points_in_pillar[1]).flatten(0, 1)
        h_ranges = torch.linspace(0.5, tpv_h-0.5, tpv_h) / tpv_h
        h_ranges = h_ranges.reshape(1, -1, 1).expand(tpv_z, -1, num_points_in_pillar[1]).flatten(0, 1)
        zh_zh = torch.stack([h_ranges, z_ranges], dim=-1) # zh, #p, 2
        # generate points for zh and level 3
        w_ranges = torch.linspace(0.5, tpv_w-0.5, num_points_in_pillar[1]) / tpv_w
        w_ranges = w_ranges.unsqueeze(0).expand(tpv_z*tpv_h, -1)
        z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(-1, 1, 1).expand(-1, tpv_h, num_points_in_pillar[1]).flatten(0, 1)
        zh_wz = torch.stack([z_ranges, w_ranges], dim=-1)

        # generate points for wz and level 1
        h_ranges = torch.linspace(0.5, tpv_h-0.5, num_points_in_pillar[0]) / tpv_h
        h_ranges = h_ranges.unsqueeze(0).expand(tpv_w*tpv_z, -1)
        w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
        wz_hw = torch.stack([w_ranges, h_ranges], dim=-1)
        # generate points for wz and level 2
        h_ranges = torch.linspace(0.5, tpv_h-0.5, num_points_in_pillar[0]) / tpv_h
        h_ranges = h_ranges.unsqueeze(0).expand(tpv_w*tpv_z, -1)
        z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(tpv_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_zh = torch.stack([h_ranges, z_ranges], dim=-1)
        # generate points for wz and level 3
        w_ranges = torch.linspace(0.5, tpv_w-0.5, tpv_w) / tpv_w
        w_ranges = w_ranges.reshape(-1, 1, 1).expand(-1, tpv_z, num_points_in_pillar[0]).flatten(0, 1)
        z_ranges = torch.linspace(0.5, tpv_z-0.5, tpv_z) / tpv_z
        z_ranges = z_ranges.reshape(1, -1, 1).expand(tpv_w, -1, num_points_in_pillar[0]).flatten(0, 1)
        wz_wz = torch.stack([z_ranges, w_ranges], dim=-1)

        reference_points = torch.cat([
            torch.stack([hw_hw, hw_zh, hw_wz], dim=1),
            torch.stack([zh_hw, zh_zh, zh_wz], dim=1),
            torch.stack([wz_hw, wz_zh, wz_wz], dim=1)
        ], dim=0) # hw+zh+wz, 3, #p, 2
        
        return reference_points


    def get_grid(self, img_metas, sampling="log",  num_pts=30, device="cuda", hn=0.5, hf=50)->(torch.tensor, torch.tensor):
        """
        img_metas: List(dict("c2w", "K", "img_shape"))
        sampling: "log" or "linear"
        num_pts: number of points per ray
        hn, hf: start and end of sampling 
        )
        """
        c2ws = []
        K = []
        img_hwc = []

        for img_meta in img_metas:
            c2ws.append(img_meta['c2w'])
            K.append(img_meta["K"])
            img_hwc.append(img_meta["img_shape"])
        c2ws = np.asarray(c2ws)
        K = np.asarray(K) 
        img_hwc = np.asarray(img_hwc)
        w = int(img_hwc[0,0,1])
        h = int(img_hwc[0,0,0])
        fl_x = K[0,0,0]
        fl_y = K[0,0,0]
        c_x = K[0,1,1]
        c_y = K[0,1,2]
        c2ws = torch.from_numpy(c2ws).to(device).float()  # (B, N, 4, 4)
        offset = c2ws.new_tensor(np.array(self.offset))
        scale = c2ws.new_tensor(np.array(self.scale))
        N =  c2ws.new_tensor(np.array([self.tpv_z, self.tpv_h, self.tpv_w]))
        intrinsics = Intrinsics(w,h,fl_x,fl_y,c_x,c_y)
        intrinsics.scale(self.intrin_factor)
            

        directions = ray_utils.get_ray_directions(intrinsics).cuda()
        uvs = ((ray_utils.create_meshgrid(int(h*self.intrin_factor),int(w*self.intrin_factor), device='cuda') + 1) / 2) #(1,116,200,2) [0->1]

        rays_os, rays_ds = [],[]
        for c2w in c2ws[0]:
            rays_o, rays_d = ray_utils.get_rays(directions, c2w[:3])
            rays_os.append(rays_o)
            rays_ds.append(rays_d)

        rays_d = torch.stack(rays_ds, dim=0) #(6,23200,3)
        rays_o = torch.stack(rays_os, dim=0) #(6,23200,3)


        if sampling == "linear":
            step =  torch.linspace(hn, hf, num_pts).to(rays_d.device)[None, ...] 
        elif sampling == "log":
            step =  torch.logspace(np.log10(hn), np.log10(hf), num_pts).to(rays_d.device)[None, ...] 
    
        points = rays_o[:,:,None,:] + rays_d[:,:,None,:] * step[None,:,:,None]

        if self.scene_contraction:
            x_grid = contract_world(points * c2ws.new_tensor(self.scene_contraction_factor)) #samples, 3
            bound_masks = (x_grid.abs() < 1.0).all(-1) # samples
        else:
            x_grid = (points - offset) / \
                    (scale * N)
            bound_masks = (x_grid.abs() < 1.0).all(-1) # samples
            

        indexes = ((x_grid /2 + 0.5 ) * N).round() # (6,23200,30,3)
        indexes = torch.clamp(indexes, N*0, N-1).int()
        grid = torch.zeros((self.tpv_z,self.tpv_h,self.tpv_w,6,2), device=points.device)
        mask = torch.zeros((self.tpv_z,self.tpv_h,self.tpv_w,6,1),dtype=bool, device=points.device)
        for i, (index, bound_mask) in enumerate(zip(indexes, bound_masks)):
            index = index[bound_mask] # (valid_ids,3)
            uv = uvs.unsqueeze(3).repeat(1,1,1,num_pts,1).view(-1,num_pts,2)[bound_mask] # (valid,2)
            grid[index[:,0], index[:,1], index[:,2],i] = uv
            mask[index[:,0], index[:,1], index[:,2],i] = True
        return  grid, mask
    

    @staticmethod
    def point_sampling(grid, mask, num_pts):
        if num_pts == 4:
            indexes = [3,6,9,12]
        elif num_pts == 32:
            indexes = [ 20,  30,  40,  50,  60,  65,  70,  73,  77,  81,  85,  89,  92,  95, 97,  99, 101, 103, 105, 108, 111, 115, 119, 123, 127, 130, 135, 140, 150, 160, 170, 180]
        else:
            indexes = torch.linspace(0,1,num_pts) * grid.size(3)

        reference_points = grid[:,:,:,indexes]
        reference_mask = mask[:,:,:,indexes]
        
        return reference_points.unsqueeze(1).flatten(2,3), reference_mask.unsqueeze(1).flatten(2,3).squeeze(-1) #(6,1,hxw,num_pts,2),(6,1,hxw,num_pts)



    @auto_fp16()
    def forward(self,
                tpv_query, # list
                key,
                value,
                *args,
                tpv_h=None,
                tpv_w=None,
                tpv_z=None,
                tpv_pos=None, # list
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            tpv_query (Tensor): Input tpv query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
        """
        output = tpv_query
        intermediate = []
        bs = tpv_query[0].shape[0]        

        if self.grid is None:
            self.grid, self.mask = self.get_grid(kwargs['img_metas'], sampling="log",  num_pts=2000, device="cuda", hn=0.1, hf=80)

        if self.ref_3d_hw_mask is None:
            self.ref_3d_hw_uvs, self.ref_3d_hw_mask = self.point_sampling(self.grid.permute(3,1,2,0,4),self.mask.permute(3,1,2,0,4),4)  # [6, 1, 40000, 4, 2])
            self.ref_3d_zh_uvs, self.ref_3d_zh_mask = self.point_sampling(self.grid.permute(3,0,1,2,4),self.mask.permute(3,0,1,2,4),32) # [6, 1, 3200,  4, 2])
            self.ref_3d_wz_uvs, self.ref_3d_wz_mask = self.point_sampling(self.grid.permute(3,1,0,2,4),self.mask.permute(3,1,0,2,4),32) # [6, 1, 3200,  4, 2])
            

        reference_points_cams = [self.ref_3d_hw_uvs, self.ref_3d_zh_uvs, self.ref_3d_wz_uvs]
        tpv_masks = [self.ref_3d_hw_mask, self.ref_3d_zh_mask, self.ref_3d_wz_mask]


        ref_cross_view = self.cross_view_ref_points.clone().unsqueeze(0).expand(bs, -1, -1, -1, -1)

        for lid, layer in enumerate(self.layers):
            output = layer(
                tpv_query,
                key,
                value,
                *args,
                tpv_pos=tpv_pos,
                ref_2d=ref_cross_view,
                tpv_h=tpv_h,
                tpv_w=tpv_w,
                tpv_z=tpv_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cams=reference_points_cams,
                tpv_masks=tpv_masks,
                **kwargs)
            tpv_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output