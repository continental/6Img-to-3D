_base_ = [
    './_base_/dataset.py',
    './_base_/optimizer.py',
    './_base_/triplane_decoder.py',
]


_dim_ = 128 # num features in triplane
num_heads = 8
_pos_dim_ = [48, 48, 32]
_ffn_dim_ = _dim_*2
_num_levels_ = 4
_num_cams_ = 6

N_h_ = 200
N_w_ = 200
N_z_ = 16

# If not contracted
offset_h = 0
offset_w = 0
offset_z = -4
offset = [offset_z, offset_h, offset_w]
scale_h = 0.25
scale_w = 0.25
scale_z = 0.25
scale = [scale_z, scale_h, scale_w]

# If contracted
scene_contraction = True
scene_contraction_factor = [0.5, 0.1,0.1]

pif = True
pif_factor = 0.125
pif_transforms = "/app/data/Town02/ClearNoon/vehicle.tesla.invisible/spawn_point_10/step_0/nuscenes/"

tpv_encoder_layers = 5
num_points_in_pillar = [4, 32, 32]
num_points = [8, 64, 64]
hybrid_attn_anchors = 16
hybrid_attn_points = 32
hybrid_attn_init = 0



self_cross_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVImageCrossAttention',
            num_cams=_num_cams_,
            deformable_attention=dict(
                type='TPVMSDeformableAttention3D',
                embed_dims=_dim_,
                num_heads=num_heads,
                num_points=num_points,
                num_z_anchors=num_points_in_pillar,
                num_levels=_num_levels_,
                floor_sampling_offset=False,
                tpv_h=N_h_,
                tpv_w=N_w_,
                tpv_z=N_z_,
            ),
            embed_dims=_dim_,
            tpv_h=N_h_,
            tpv_w=N_w_,
            tpv_z=N_z_,
        ),
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=N_h_,
            tpv_w=N_w_,
            tpv_z=N_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
        ),
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('cross_attn', 'norm','self_attn','norm', 'ffn', 'norm')
)

self_layer = dict(
    type='TPVFormerLayer',
    attn_cfgs=[
        dict(
            type='TPVCrossViewHybridAttention',
            tpv_h=N_h_,
            tpv_w=N_w_,
            tpv_z=N_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
        )
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=('self_attn', 'norm', 'ffn', 'norm')
)


model = dict(
    type='TPVFormer',
    output_features=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    tpv_head=dict(
        type='TPVFormerHead',
        tpv_h=N_h_,
        tpv_w=N_w_,
        tpv_z=N_z_,
        num_feature_levels=_num_levels_,
        num_cams=_num_cams_,
        embed_dims=_dim_,
        encoder=dict(
            type='TPVFormerEncoder',
            tpv_h=N_h_,
            tpv_w=N_w_,
            tpv_z=N_z_,
            offset = [offset_z, offset_h, offset_w],
            scale = [scale_z, scale_h, scale_w],
            intrin_factor=pif_factor,
            scene_contraction=scene_contraction,
            scene_contraction_factor=scene_contraction_factor,
            num_layers=tpv_encoder_layers,
            num_points_in_pillar=num_points_in_pillar,
            num_points_in_pillar_cross_view=[16, 16, 16],
            return_intermediate=False,
            transformerlayers=[
                self_cross_layer,
                self_cross_layer,
                self_cross_layer,
                self_layer,
                self_layer,
            ]),
        positional_encoding=dict(
            type='CustomPositionalEncoding',
            num_feats=_pos_dim_,
            h=N_h_,
            w=N_w_,
            z=N_z_
        )))