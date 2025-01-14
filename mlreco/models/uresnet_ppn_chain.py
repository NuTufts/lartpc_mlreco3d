import numpy as np
import torch
import torch.nn as nn
import time

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.models.layers.common.ppnplus import PPN, PPNLonelyLoss
from mlreco.models.layers.common.kpscorenet import KeypointScoreNet, KeypointScoreNetLoss
from mlreco.models.uresnet import SegmentationLoss
from collections import defaultdict
from mlreco.models.uresnet import UResNet_Chain

class UResNetPPN(nn.Module):
    """
    A model made of UResNet backbone and PPN layers. Typical configuration:

    .. code-block:: yaml

        model:
          name: uresnet_ppn_chain
          modules:
            uresnet_lonely:
              # Your uresnet config here
            ppn:
              # Your ppn config here

    Configuration
    -------------
    net_type: str
        The type of keypoint proposal network. Options ['ppn','kpscorenet']. Default: 'ppn'
    data_dim: int, default 3
    num_input: int, default 1
    allow_bias: bool, default False
    spatial_size: int, default 512
    leakiness: float, default 0.33
    activation: dict
        For activation function, defaults to `{'name': 'lrelu', 'args': {}}`
    norm_layer: dict
        For normalization function, defaults to `{'name': 'batch_norm', 'args': {}}`

    depth: int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters: int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps: int, default 2
        Convolution block repetition factor
    input_kernel: int, default 3
        Receptive field size for very first convolution after input layer.

    num_classes: int, default 5
    score_threshold: float, default 0.5
    classify_endpoints: bool, default False
        Enable classification of points into start vs end points.
    ppn_resolution: float, default 1.0
    ghost: bool, default False
    downsample_ghost: bool, default True
    use_true_ghost_mask: bool, default False
    mask_loss_name: str, default 'BCE'
        Can be 'BCE' or 'LogDice'
    particles_label_seg_col: int, default -2
        Which column corresponds to particles' semantic label
    track_label: int, default 1

    See Also
    --------
    mlreco.models.uresnet.UResNet_Chain, mlreco.models.layers.common.ppnplus.PPN
    """
    MODULES = ['mink_uresnet', 'mink_uresnet_ppn_chain', 'mink_ppn']

    def __init__(self, cfg):
        super(UResNetPPN, self).__init__()
        self.model_config = cfg
        self.ghost = cfg.get('uresnet_lonely', {}).get('ghost', False)
        assert self.ghost == cfg.get('ppn', {}).get('ghost', False)
        self.backbone = UResNet_Chain(cfg)

        self._kpnet_type = cfg.get('ppn',{}).get('net_type','ppn')
        if self._kpnet_type=='ppn':
            self.ppn = PPN(cfg)
        elif self._kpnet_type=='kpscorenet':
            self.ppn = KeypointScoreNet(cfg)
        else:
            raise ValueError("Unrecognized keypoint network type: ",self._kpnet_type)
        
        self.num_classes = self.backbone.num_classes
        self.num_filters = self.backbone.F
        self.segmentation = ME.MinkowskiLinear(
            self.num_filters, self.num_classes)

    def forward(self, input):

        labels = None

        if len(input) == 1:
            # PPN without true ghost mask propagation
            input_tensors = [input[0]]
        elif len(input) == 2:
            # PPN with true ghost mask propagation
            input_tensors = [input[0]]
            labels = input[1]

        out = defaultdict(list)

        for igpu, x in enumerate(input_tensors):
            # input_data = x[:, :5]

            # run uresnet (encoder + class/ghost decoder)
            res = self.backbone([x])
            out.update({'ghost': res['ghost']})

            # now run PPN
            if self._kpnet_type=="ppn":
                if self.ghost:
                    if self.ppn.use_true_ghost_mask:
                        res_ppn = self.ppn(res['finalTensor'][igpu],
                                           res['decoderTensors'][igpu],
                                           ghost=res['ghost_sptensor'][igpu],
                                           ghost_labels=labels)
                    else:
                        res_ppn = self.ppn(res['finalTensor'][igpu],
                                           res['decoderTensors'][igpu],
                                           ghost=res['ghost_sptensor'][igpu])

                else:
                    res_ppn = self.ppn(res['finalTensor'][igpu],
                                       res['decoderTensors'][igpu])
            elif self._kpnet_type=="kpscorenet":
                """ Run KeypointScoreNetwork 
                  Attempting to run with same signatures as PPN
                """
                if self.ghost:
                    if self.ppn.use_true_ghost_mask:
                        res_ppn = self.ppn(res['finalTensor'][igpu],
                                           res['decoderTensors'][igpu],
                                           ghost=res['ghost_sptensor'][igpu],
                                           ghost_labels=labels)
                    else:
                        res_ppn = self.ppn(res['finalTensor'][igpu],
                                           res['decoderTensors'][igpu],
                                           ghost=res['ghost_sptensor'][igpu])

                else:
                    res_ppn = self.ppn(res['finalTensor'][igpu],
                                       res['decoderTensors'][igpu])
                
            # if self.training:
            #     res_ppn = self.ppn(res['finalTensor'], res['encoderTensors'], particles_label)
            # else:
            #     res_ppn = self.ppn(res['finalTensor'], res['encoderTensors'])
            segmentation = self.segmentation(res['decoderTensors'][igpu][-1])
            out['segmentation'].append(segmentation.F)

            if self._kpnet_type=="ppn":
                out['points'].append(res_ppn['points'])
                out['mask_ppn'].append(res_ppn['mask_ppn'])
                out['ppn_layers'].append(res_ppn['ppn_layers'])
                out['ppn_coords'].append(res_ppn['ppn_coords'])
            elif self._kpnet_type=="kpscorenet":
                out['ppn_kpscore'].append(res_ppn['ppn_kpscore'][igpu].F)

        return out


class UResNetPPNLoss(nn.Module):
    """
    See Also
    --------
    mlreco.models.uresnet.SegmentationLoss, mlreco.models.layers.common.ppnplus.PPNLonelyLoss
    """
    def __init__(self, cfg):
        super(UResNetPPNLoss, self).__init__()

        self._kpnet_type = cfg.get('ppn',{}).get('net_type','ppn')
        if self._kpnet_type=='ppn':
            self.ppn_loss = PPNLonelyLoss(cfg)
        elif self._kpnet_type=='kpscorenet':
            self.ppn_loss = KeypointScoreNetLoss(cfg)
        else:
            raise ValueError("Unrecognized keypoint network type: ",self._kpnet_type)
        
        self.segmentation_loss = SegmentationLoss(cfg)

    def forward(self, outputs, segment_label, particles_label, weights=None):

        res_segmentation = self.segmentation_loss(
            outputs, segment_label, weights=weights)

        res_ppn = self.ppn_loss(
            outputs, segment_label, particles_label)

        res = {
            'loss': res_segmentation['loss'] + res_ppn['loss'],
            'accuracy': (res_segmentation['accuracy'] + res_ppn['accuracy'])/2
        }

        res.update({'segmentation_'+k:v for k, v in res_segmentation.items()})
        res.update({'ppn_'+k:v for k, v in res_ppn.items()})

        return res
