import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmengine.registry import MODELS
import pdb
import sys


@MODELS.register_module()
class BEVFuser(nn.Module):

    def __init__(self):
        super().__init__()
        norm_cfg = dict(type='SyncBN', eps=1e-3, momentum=0.01)
        self.conv = nn.Sequential(
            # fuse
            ConvModule(256, 256, 3, 1, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(256, 256, 3, 1, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(256, 256, 3, 1, 1, bias=False, norm_cfg=norm_cfg),

            ConvModule(256, 512, 3, 2, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(512, 512, 3, 1, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(512, 512, 3, 1, 1, bias=False, norm_cfg=norm_cfg),

            ConvModule(512, 1024, 3, 2, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(1024, 1024, 3, 1, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(1024, 1024, 3, 1, 1, bias=False, norm_cfg=norm_cfg),

            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )

    def forward(self, stack_feats):
        pdb.set_trace()
        # tuple_size = sys.getsizeof(stack_feats)
        B, C, H, W = stack_feats.size()
        prev_feats, this_feats = torch.split(stack_feats, B // 2, 0)
        feats = torch.cat((prev_feats, this_feats), 1)

        return self.conv(feats)

@MODELS.register_module()
class BEVMultiFuser(nn.Module):

    def __init__(self):
        super().__init__()
        norm_cfg = dict(type='SyncBN', eps=1e-3, momentum=0.01)
        self.conv = nn.Sequential(
            # fuse
            # ConvModule(256, 256, 3, 1, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(1024, 256, 3, 1, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(256, 256, 3, 1, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(256, 256, 3, 1, 1, bias=False, norm_cfg=norm_cfg),

            ConvModule(256, 512, 3, 2, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(512, 512, 3, 1, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(512, 512, 3, 1, 1, bias=False, norm_cfg=norm_cfg),

            ConvModule(512, 1024, 3, 2, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(1024, 1024, 3, 1, 1, bias=False, norm_cfg=norm_cfg),
            ConvModule(1024, 1024, 3, 1, 1, bias=False, norm_cfg=norm_cfg),

            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )

    def forward(self, stack_feats):
        # pdb.set_trace()
        fused_feats = []
        #stack_feats是attention.py中输出的bev_output_feat_maps, atten_maps
        (feats,atten)=stack_feats
        #这里将bev_output_feat_maps赋值给feats，feats由多个tensor组成，所以用循环。
        #feat为bev_output_feat_maps中的tensor
        for feat in feats:
            # 取feats中的每个feat,并将B/2，沿C维度拼接
            B, C, H, W = feat.size()#(256, 256, 16, 16)
            prev_feats, this_feats = torch.split(feat, B // 2, 0)#维度均为[128, 256, 16, 16]
            fused_feat= torch.cat((prev_feats, this_feats), 1)#[128, 512, 16, 16]
            fused_feats.append(fused_feat)
        #最后将fused_feats沿C维度拼接
        out_feats = torch.cat(fused_feats, dim=1)#[128, 1024, 16, 16]
        return self.conv(out_feats)
