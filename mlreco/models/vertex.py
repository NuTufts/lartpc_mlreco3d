
import numpy as np
import torch
import torch.nn as nn
import time

import MinkowskiEngine as ME
import MinkowskiFunctional as MF
from mlreco.models.uresnet import SegmentationLoss
from collections import defaultdict
from mlreco.models.uresnet import UResNet_Chain
from mlreco.models.layers.common.vertex_ppn import VertexPPN, VertexPPNLoss

class VertexPPNChain(nn.Module):
    """
    Experimental model for PPN-like vertex prediction
    """
    MODULES = ['mink_uresnet', 'mink_uresnet_ppn_chain', 'mink_ppn']

    def __init__(self, cfg):
        super(VertexPPNChain, self).__init__()
        self.model_config = cfg
        self.backbone = UResNet_Chain(cfg)
        self.vertex_ppn = VertexPPN(cfg)
        self.num_classes = self.backbone.num_classes
        self.num_filters = self.backbone.F
        self.segmentation = ME.MinkowskiLinear(
            self.num_filters, self.num_classes)

    def forward(self, input):

        primary_labels = None
        if self.training:
            assert(len(input) == 2)
            primary_labels = input[1][:, -2]
            segment_labels = input[1][:, -1]

        input_tensors = [input[0][:, :5]]

        out = defaultdict(list)

        for igpu, x in enumerate(input_tensors):
            # input_data = x[:, :5]
            res = self.backbone([x])
            input_sparse_tensor = res['encoderTensors'][0][0]
            segmentation = self.segmentation(res['decoderTensors'][igpu][-1])
            res_vertex = self.vertex_ppn(res['finalTensor'][igpu],
                               res['decoderTensors'][igpu],
                               input_sparse_tensor=input_sparse_tensor,
                               primary_labels=primary_labels,
                               segment_labels=segment_labels)
            out['segmentation'].append(segmentation.F)
            out.update(res_vertex)
            

        return out


class UResNetVertexLoss(nn.Module):
    """
    See Also
    --------
    mlreco.models.uresnet.SegmentationLoss, mlreco.models.layers.common.ppnplus.PPNLonelyLoss
    """
    def __init__(self, cfg):
        super(UResNetVertexLoss, self).__init__()
        self.vertex_loss = VertexPPNLoss(cfg)
        self.segmentation_loss = SegmentationLoss(cfg)

    def forward(self, outputs, kinematics_label):

        res_segmentation = self.segmentation_loss(outputs, kinematics_label)

        res_vertex = self.vertex_loss(outputs, kinematics_label)

        res = {
            'loss': res_segmentation['loss'] + res_vertex['vertex_loss'],
            'accuracy': (res_segmentation['accuracy'] + res_vertex['vertex_acc']) / 2.0,
            'reg_loss': res_vertex['vertex_reg_loss']
        }
        return res
