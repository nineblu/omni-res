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

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np


def mask_iou(mask1, mask2, accuracy_thresholds):
    """
    :param mask1:  l
    :param mask2:  l
    :return: iou
    """
    mask1 = mask1.reshape([-1])
    mask2 = mask2.reshape([-1])
    t = mask1 > 0.5
    p = mask2 > 0.5
    intersection = np.logical_and(t, p)
    union = np.logical_or(t, p)
    iou = (np.sum(intersection > 0) + 1e-20) / (np.sum(union > 0) + 1e-20)

    I = intersection.sum()
    U = union.sum()
    if I==0 or U==0:
        iou = 0.

    ap = dict()
    for thresh in accuracy_thresholds:
        ap[thresh] = int(iou > thresh)
    return iou, ap, I, U


def mask_processing(mask,info_img):
    h, w, nh, nw, dx, dy, _ = info_img
    mask = mask[dy:dy + nh, dx:dx + nw,None]
    mask = cv2.resize(mask,(int(w),int(h)))
    return mask