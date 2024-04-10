"""
This code is almost completely copied from TPVFormer https://github.com/wzzheng/TPVFormer


Please find the Apache 2 license conditions here:

https://github.com/wzzheng/TPVFormer/blob/a1cf223ae4b79f56a2b046016c35a8fb3a0b6284/LICENSE
"""



from mmcv.runner import force_fp32, auto_fp16, BaseModule
from mmseg.models import SEGMENTORS, builder
import warnings


@SEGMENTORS.register_module(force=True)
class TPVFormer(BaseModule):

    def __init__(self,
                 img_backbone=None,
                 img_neck=None,
                 tpv_head=None,
                 pretrained=None,
                 output_features=False,
                 **kwargs,
                 ):

        super().__init__()

        if tpv_head:
            self.tpv_head = builder.build_head(tpv_head)
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck:
            self.img_neck = builder.build_neck(img_neck)


        if pretrained is None:
            img_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')

        if img_backbone:
            if img_pretrained is not None:
                warnings.warn('DeprecationWarning: pretrained is a deprecated \
                    key, please consider using init_cfg')
                self.img_backbone.init_cfg = dict(
                    type='Pretrained', checkpoint=img_pretrained)

        self.fp16_enabled = False
        self.output_features = output_features

    @auto_fp16(apply_to=('img'))
    def extract_img_feat(self, img):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values()) # [torch.Size([6, 512, 116, 200]), torch.Size([6, 1024, 58, 100]), torch.Size([6, 2048, 29, 50])]
        else:
            return None
        if hasattr(self, 'img_neck'):
            img_feats = self.img_neck(img_feats) # [torch.Size([6, 128, 116, 200]), torch.Size([6, 128, 58, 100]), torch.Size([6, 128, 29, 50]), torch.Size([6, 128, 15, 25])]

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped # [torch.Size([1, 6, 128, 116, 200]), torch.Size(1, [6, 128, 58, 100]), torch.Size([1, 6, 128, 29, 50]), torch.Size([1, 6, 128, 15, 25])]

    @auto_fp16(apply_to=('img', 'points'))
    def forward(self,
                img_metas=None,
                img=None,
        ):
        """Forward training function.
        """
        img_feats = self.extract_img_feat(img=img)
        triplane = self.tpv_head(img_feats, img_metas) # [torch.Size([1, 40000, 128]), torch.Size([1, 3200, 128]), torch.Size([1, 3200, 128])]
        if self.output_features:
            return triplane, img_feats
        return triplane