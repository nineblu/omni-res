# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from omni_res.layers.aspp import aspp_decoder
from ..utils.box_op import bboxes_iou


class MCNhead(nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    """
    def __init__(
        self, 
        hidden_size=512, 
        anchors=[[137, 256], [248, 272], [386, 271]], 
        arch_mask=[[0, 1, 2]], 
        layer_no=0, 
        in_ch=512, 
        n_classes=0, 
        ignore_thre=0.5
    ):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """

        super(MCNhead, self).__init__()
        self.anchors = anchors
        self.anch_mask = arch_mask[layer_no]
        self.n_anchors = len(self.anch_mask)
        self.n_classes = n_classes
        self.ignore_thre = ignore_thre
        self.l2_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.stride = 32 # strides[layer_no]
        self.all_anchors_grid = [(w / self.stride, h / self.stride) for w, h in self.anchors]
        self.masked_anchors = [self.all_anchors_grid[i] for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)
        self.d_proj = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        self.s_proj = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)

        self.dconv = nn.Conv2d(in_channels=in_ch, out_channels=self.n_anchors * (self.n_classes + 5), kernel_size=1, stride=1, padding=0)
        self.sconv = nn.Sequential(aspp_decoder(in_ch, hidden_size//2, 1), nn.UpsamplingBilinear2d(scale_factor=8))

    def forward(self, seg_in):
        mask = self.sconv(seg_in)
        mask_logit = mask.squeeze(1).sigmoid()
        return mask_logit, mask
