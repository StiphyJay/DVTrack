# Copyright (c) Gorilla-Lab. All rights reserved.

import numpy as np
import torch
import pdb
from torch import nn
from torch.nn import BatchNorm1d
from torch.nn import functional as F
import spconv.pytorch as spconv
from spconv.pytorch import SparseConv3d, SubMConv3d

# from models.utils import Empty, change_default_args
# from models.torchie.cnn import constant_init, kaiming_init
# from models.torchie.trainer import load_checkpoint

from .scn import SparseBasicBlock
# from .. import builder
# from ..registry import BACKBONES
from models.utils import build_conv_layer, build_norm_layer


# @BACKBONES.register_module
# class MultiViewBackbone(nn.Module):  # Small Depth
class MultiViewEncoder(nn.Module):  # Small Depth
    def __init__(
        # self, num_input_features=128, norm_cfg=None, name="MultiViewBackbone", **kwargs
        self, num_input_features=128, norm_cfg=None, name="MultiViewEncoder", **kwargs
    ):
        super().__init__()
        self.name = name
        
        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-5, momentum=0.1)
        # input: # [1600, 1200, 41]
        self.middle_conv = spconv.SparseSequential(
            # SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            # build_norm_layer(norm_cfg, 16)[1],
            # nn.ReLU(),
            # SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            # SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseConv3d(
                64, 128, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )
        self.bev_conv = spconv.SparseSequential(
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )
        self.rv_conv = spconv.SparseSequential(
            SparseConv3d(
                128, 128, (1, 1, 3), (1, 1, 2), (0, 0, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res4"),
            SparseConv3d(
                128, 128, (1, 1, 3), (1, 1, 2), (0, 0, 1), bias=False
            ),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res5"),
            SparseConv3d(
                128, 128, (1, 1, 3), (1, 1, 2), (0, 0, 1), bias=False
            ),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

    def forward(self, ret):
        
        # input: # [41, 1600, 1408] #[256,16,21,128,128]
        # print('multiview-->ret>pconv.SparseConvTensor==', ret) 
        ##print('ret>pconv.SparseConvTensorshape==', ret.shape)#卷积没有shape
        ret = self.middle_conv(ret)#256*128*2*16*16
        # print('ret>self.middle_conv(ret)==', ret) 
        # print('ret>self.middle_conv(ret).shape==', ret.shape)
        # ret_bev = self.bev_conv(ret)
        # print('bev->self.bev_conv(ret)==', ret_bev)
        # print('bev->self.bev_conv(ret).shape==', ret_bev.shape)
        ret_bev = ret
        ret_bev = ret_bev.dense()
        # print('bev>ret_bev.dense() ==', ret_bev)
        # print('bev>ret_bev.dense().shape==', ret_bev.shape)//
        N, C, D, H, W = ret_bev.shape
        # print('bev=->N, C, D, H, W = ==', N, C, D, H, W )
        ret_bev = ret_bev.view(N, C * D, H, W)
        # print('ret_bev.view(N, C * D, H, W).shape==', ret_bev.shape)//
        # print('=ret_bev ==', ret_bev )

        ret_rv = self.rv_conv(ret)
        # print('rv>self.rv_conv(ret)==', ret_rv)
        # print('rv>self.rv_conv(ret).shape==', ret_rv.shape)

        ret_rv = ret_rv.dense()
        # print('rv=ret_rv.dense() ==', ret_rv)
        # print('rv=ret_rv.dense().shape==', ret_rv.shape)//
        N, C, D, H, W = ret_rv.shape
        # print('rv->N, C, D, H, W = ==', N, C, D, H, W )
        ret_rv = ret_rv.permute(0, 1, 4, 3, 2)
        ret_rv = ret_rv.contiguous().view(N, C * W, H, D)
        # print('rv=ret_rv.contiguous().view(N, C * W, H, D).shape==', ret_rv.shape)//
        # print('=ret_rv.shape ==', ret_rv.shape )
        return ret_bev, ret_rv #256 256 16 16/ 256 256 16 2