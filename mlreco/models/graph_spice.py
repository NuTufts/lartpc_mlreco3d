import torch
import numpy as np

from .cluster_cnn.losses.spatial_embeddings import *
from .cluster_cnn import cluster_model_construct, backbone_construct, spice_loss_construct

from pprint import pprint

class GraphSPICE(nn.Module):
    '''
    Neighbor-graph embedding based particle clustering
    '''

    def __init__(self, cfg, name='spice'):
        super(GraphSPICE, self).__init__()
        self.model_name = cfg[name]['spice_model_name']
        self.skip_classes = cfg[name].get('skip_classes', [2, 3, 4])
        self.model_config = cfg[name]['spice_model_config']
        self.dimension = cfg[name].get('dimension', 3)
        self.model = cluster_model_construct(self.model_config, self.model_name)

    def _forward(self, input):

        point_cloud, cluster_label = input

        mask = ~np.isin(cluster_label[:, -1].detach().cpu().numpy(), self.skip_classes)
        x = [point_cloud[mask], cluster_label[mask]]

        res = self.model(x)
        return res

    def forward(self, input):
        '''

        '''
        if self.training:
            res = self._forward(input)
        else:
            point_cloud, segment_label = input

            mask = ~np.isin(segment_label[:, -1].detach().cpu().numpy(), self.skip_classes)
            x = [point_cloud[mask], segment_label[mask]]
            res = self.model(x)
        return res

class GraphSPICELoss(nn.Module):

    def __init__(self, cfg, name='spice_loss'):
        super(GraphSPICELoss, self).__init__()
        self.loss_config = cfg[name]
        self.loss_name = cfg[name]['name']
        self.skip_classes = cfg[name].get('skip_classes', [2, 3, 4])
        self.loss_fn = spice_loss_construct(self.loss_name)(self.loss_config)
        # print("LOSS FN = ", self.loss_fn)

    def forward(self, result, segment_label, cluster_label):
        '''

        '''
        if len(self.skip_classes) != 0:
            mask = ~np.isin(segment_label[0][:, -1].detach().cpu().numpy(), self.skip_classes)
            segment_label = [segment_label[0][mask]]
            cluster_label = [cluster_label[0][mask]]
        res = self.loss_fn(result, segment_label, cluster_label)
        return res