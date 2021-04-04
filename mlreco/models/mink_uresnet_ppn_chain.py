import numpy as np
import torch
import torch.nn as nn
import time

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.mink.layers.ppnplus import PPN, PPNLonelyLoss
from mlreco.models.mink_uresnet import SegmentationLoss
from mlreco.mink.layers.uresnet import UResNet
from collections import defaultdict


class UResNetPPN(nn.Module):

    def __init__(self, cfg, name='uresnet_chain'):
        super(UResNetPPN, self).__init__()
        self.backbone = UResNet(cfg)
        self.ppn = PPN(cfg)
        self.model_config = cfg[name]
        self.num_classes = self.model_config.get('num_classes', 5)
        self.num_filters = self.model_config.get('num_filters', 16)
        self.segmentation = ME.MinkowskiLinear(
            self.num_filters, self.num_classes)

    def forward(self, input):
        device = input[0].device
        out = defaultdict(list)
        for igpu, x in enumerate(input):
            input_data = x[:, :5]
            res = self.backbone(input_data)
            res_ppn = self.ppn(res['finalTensor'], res['encoderTensors'])
            segmentation = self.segmentation(res['decoderTensors'][-1])
            out['segmentation'].append(segmentation.F)
            out['points'].append(res_ppn['points'])
            out['mask_ppn'].append(res_ppn['mask_ppn'])
            out['ppn_layers'].append(res_ppn['ppn_layers'])
            out['ppn_coords'].append(res_ppn['ppn_coords'])
            
        for t in out['mask_ppn'][0]:
            print(t.shape)
        return out


class UResNetPPNLoss(nn.Module):

    def __init__(self, cfg, name='mink_uresnet_ppn_loss'):
        super(UResNetPPNLoss, self).__init__()
        self.ppn_loss = PPNLonelyLoss(cfg)
        self.segmentation_loss = SegmentationLoss(cfg)

    def forward(self, outputs, segment_label, particles_label, weight=None):

        res_segmentation = self.segmentation_loss(
            outputs, segment_label, weight=weight)

        res_ppn = self.ppn_loss(
            outputs, segment_label, particles_label)

        res = {
            'loss': res_segmentation['loss'] + res_ppn['loss'],
            'accuracy': (res_segmentation['accuracy'] + res_ppn['accuracy']) / 2.0
        }
        return res