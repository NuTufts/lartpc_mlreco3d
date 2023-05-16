import numpy as np
import torch
import torch.nn as nn

import MinkowskiEngine as ME
import MinkowskiFunctional as MF

from mlreco.utils import local_cdist
from mlreco.models.layers.common.blocks import ResNetBlock, SPP, ASPP
from mlreco.models.layers.common.activation_normalization_factories import activations_construct
from mlreco.models.layers.common.configuration import setup_cnn_configuration
from mlreco.models.layers.common.extract_feature_map import MinkGhostMask

from collections import Counter

from mlreco.models.layers.cluster_cnn.losses.misc import BinaryCELogDiceLoss


class AttentionMask(torch.nn.Module):
    '''
    Returns a masked tensor of x according to mask, where the number of
    coordinates between x and mask differ
    '''
    def __init__(self, score_threshold=0.5):
        super(AttentionMask, self).__init__()
        self.prune = ME.MinkowskiPruning()
        self.score_threshold=score_threshold

    def forward(self, x, mask):

        assert x.tensor_stride == mask.tensor_stride

        device = x.F.device
        # Create a mask sparse tensor in x-coordinates
        x0 = ME.SparseTensor(
            coordinates=x.C,
            features=torch.zeros(x.F.shape[0], mask.F.shape[1]).to(device),
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride)

        mask_in_xcoords = x0 + mask

        x_expanded = ME.SparseTensor(
            coordinates=mask_in_xcoords.C,
            features=torch.zeros(mask_in_xcoords.F.shape[0],
                                 x.F.shape[1]).to(device),
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride)

        x_expanded = x_expanded + x

        target = mask_in_xcoords.F.int().bool().squeeze()
        x_pruned = self.prune(x_expanded, target)
        return x_pruned


class MergeConcat(torch.nn.Module):

    def __init__(self):
        super(MergeConcat, self).__init__()

    def forward(self, input, other):

        assert input.tensor_stride == other.tensor_stride
        device = input.F.device

        # Create a placeholder tensor with input.C coordinates
        x0 = ME.SparseTensor(
            coordinates=input.C,
            features=torch.zeros(input.F.shape[0], other.F.shape[1]).to(device),
            coordinate_manager=input.coordinate_manager,
            tensor_stride=input.tensor_stride)

        # Set placeholder values with other.F features by performing
        # sparse tensor addition.
        x1 = x0 + other

        # Same procedure, but with other
        x_expanded = ME.SparseTensor(
            coordinates=x1.C,
            features=torch.zeros(x1.F.shape[0],
                                 input.F.shape[1]).to(device),
            coordinate_manager=input.coordinate_manager,
            tensor_stride=input.tensor_stride)

        x2 = x_expanded + input

        # Now input and other share the same coordinates and shape
        concated = ME.cat(x1, x2)
        return concated


class ExpandAs(nn.Module):
    def __init__(self):
        super(ExpandAs, self).__init__()

    def forward(self, x, shape, labels=None, 
                propagate_all=False,
                use_binary_mask=False):
        '''
            x: feature tensor of input sparse tensor (N x F)
            labels: N x 0 tensor of labels
            propagate_all: If True, PPN will not perform masking at each layer.

        '''
        device = x.F.device
        features = x.F
        if labels is not None:
            assert labels.shape[0] == x.F.shape[0]
            features[labels] = 1.0
        if propagate_all:
            features[None] = 1.0
        if use_binary_mask:
            features = (features > 0.5).float().expand(*shape)
        else:
            features = features.expand(*shape)
        # if labels is not None:
        #     features_expand = features.expand(*shape).clone()
        #     features_expand[labels] = 1.0
        # else:
        #     features_expand = features.expand(*shape)
        output = ME.SparseTensor(
            features=features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)
        return output


def get_ppn_weights(d_positives, mode='const', eps=1e-3):

    device = d_positives.device

    num_positives = d_positives.sum()
    num_negatives = d_positives.nelement() - num_positives

    w = num_positives.float() / \
        (num_positives + num_negatives).float()

    weight_ppn = torch.ones(d_positives.shape[0]).to(device)

    if mode == 'const':
        weight_ppn[d_positives] = 1-w
        weight_ppn[~d_positives] = w
    elif mode == 'log':
        weight_ppn[d_positives] = -torch.log(w + eps)
        weight_ppn[~d_positives] = -torch.log(1-w + eps)
    elif mode == 'sqrt':
        weight_ppn[d_positives] = torch.sqrt(w + eps)
        weight_ppn[~d_positives] = torch.sqrt(1-w + eps)
    else:
        raise ValueError("Weight mode {} not supported!".format(mode))

    return weight_ppn


class KeypointScoreNet(torch.nn.Module):
    '''
    Keypoint Proposal Network that predicts a score for each voxel based on the nearest keypoint.

    It requires a UResNet network as a backbone.

    Configuration
    -------------
    ghost: bool
      If true, we will mask out ghost voxels before calculating features
    use_true_ghost_mask: bool
      If true and ghost==true, then we use truth ghost labels for mask.
      If false and ghost==true, then we use ghost predictions for mask
    reps: int
      Number of (non-downsampling) ResNet blocks per layer. Default: 2
    depth: int
      Number of layers. Default: 5
    num_classes: int
      Number of keypoint classes: 6
    num_filters: int
      Number of kernels per convolutional layer
    propagate_all: bool
        If True, PPN will not perform masking at each layer. Default: False

    Output
    ------

    See Also
    --------
    PPNLonelyLoss, mlreco.models.uresnet_ppn_chain
    '''

    # CLASS DEFINITIONS
    NUM_KP_CLASSES = 6
    KP_CLASS_NAMES=["kp_nu",
                    "kp_trackstart",
                    "kp_trackend",
                    "kp_shower",
                    "kp_michel",
                    "kp_delta"]
    
    
    def __init__(self, cfg, name='kpscore'):
        super(KeypointScoreNet, self).__init__()
        setup_cnn_configuration(self, cfg, name)

        self.model_cfg = cfg.get('ppn', {})
        # UResNet Configurations
        self.reps = self.model_cfg.get('reps', 2)
        self.depth = self.model_cfg.get('depth', 5)
        self.num_classes = KeypointScoreNet.NUM_KP_CLASSES
        self.num_filters = self.model_cfg.get('filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        self.use_true_ghost_mask = self.model_cfg.get('use_true_ghost_mask',False)

        # Initialize Decoder
        self.decoding_block = []
        self.decoding_conv = []
        #self.ppn_pred = nn.ModuleList() # used by PPN for attention-type mechanism
        for i in range(self.depth-2, -1, -1):
            m = []
            m.append(ME.MinkowskiBatchNorm(self.nPlanes[i+1]))
            m.append(activations_construct(
                self.activation_name, **self.activation_args))
            m.append(ME.MinkowskiConvolutionTranspose(
                in_channels=self.nPlanes[i+1],
                out_channels=self.nPlanes[i],
                kernel_size=2,
                stride=2,
                dimension=self.D))
            m = nn.Sequential(*m)
            self.decoding_conv.append(m)
            m = []
            for j in range(self.reps):
                m.append(ResNetBlock(self.nPlanes[i] * (2 if j == 0 else 1),
                                     self.nPlanes[i],
                                     dimension=self.D,
                                     activation=self.activation_name,
                                     activation_args=self.activation_args))
            m = nn.Sequential(*m)
            self.decoding_block.append(m)
            #self.ppn_pred.append(ME.MinkowskiLinear(self.nPlanes[i], 1))
        self.decoding_block = nn.Sequential(*self.decoding_block)
        self.decoding_conv = nn.Sequential(*self.decoding_conv)

        self.sigmoid = ME.MinkowskiSigmoid()
        #self.expand_as = ExpandAs()
        #self.propagate_all = self.model_cfg.get('propagate_all', False)

        self.final_block = ResNetBlock(self.nPlanes[0],
                                       self.nPlanes[0],
                                       dimension=self.D,
                                       activation=self.activation_name,
                                       activation_args=self.activation_args)

        self.kpscore_pred = ME.MinkowskiConvolution(self.nPlanes[0],
                                                    self.num_classes,
                                                    kernel_size=1,
                                                    stride=1,
                                                    dimension=self.D)
        
        # Ghost point removal options
        self.ghost = self.model_cfg.get('ghost', False)

        self.masker = AttentionMask()
        self.merge_concat = MergeConcat()

        if self.ghost:
            #print("Ghost Masking is enabled for MinkPPN.")
            self.ghost_mask = MinkGhostMask(self.D)
            self.use_true_ghost_mask = self.model_cfg.get(
                'use_true_ghost_mask', False)
            self.downsample_ghost = self.model_cfg.get('downsample_ghost', True)

        print('Total Number of Trainable Parameters (mink_kpscore)= {}'.format(
            sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, final, decoderTensors, ghost=None, ghost_labels=None):
        """
        inputs
        ------

        We use the output of module UResNet in 'mlreco.model.layers.common.uresnet_layers.py'
        It produces: { 'encoderTensors':output of ResNet blocks in a layer before the strided convolution,
                       'decoderTensors':output of ResNet blocks in a lyaer before upsampling convolution,
                       'finalTensor':feature tensor at deepest layer,
                       'features_ppn':list of intermediate tensors (right after encoding block + convolution) }
        decoderTensors that is passed is 'decoderTensors' from UResnet.
        first tensor is deepest layer, last is at input resolution, i.e.  the output final resolution.
        """
        
        ppn_layers, ppn_coords = [], []
        tmp = []
        mask_ppn = []
        device = final.device

        # We need to make labels on-the-fly to include true points in the
        # propagated masks during training

        decoder_feature_maps = []

        # prepare the ghost masks
        if self.ghost:
            # Downsample stride 1 ghost mask to all intermediate decoder layers
            with torch.no_grad():
                if self.use_true_ghost_mask:
                    assert ghost_labels is not None
                    # TODO: Not sure what's going on here
                    ghost_mask_tensor = ghost_labels[:, -1] < self.num_classes
                    ghost_coords = ghost_labels[:, :4]
                else:
                    ghost_mask_tensor = 1.0 - torch.argmax(ghost.F,
                                                           dim=1,
                                                           keepdim=True)
                    ghost_coords = ghost.C
                    ghost_coords_man = final.coordinate_manager
                    ghost_tensor_stride = ghost.tensor_stride
                ghost_mask = ME.SparseTensor(
                    features=ghost_mask_tensor,
                    coordinates=ghost_coords,
                    coordinate_manager=ghost_coords_man,
                    tensor_stride=ghost_tensor_stride)

            for t in decoderTensors[::-1]:
                scaled_ghost_mask = self.ghost_mask(ghost_mask, t)
                nonghost_tensor = self.masker(t, scaled_ghost_mask)
                decoder_feature_maps.append(nonghost_tensor)

            decoder_feature_maps = decoder_feature_maps[::-1] # throw away the last entry?

        else:
            decoder_feature_maps = decoderTensors

        # apply the layers

        # start with the output of the UResnet at the deepest layer
        x = final
        #print("[kpscorenet] input=",x.F.shape)

        for i, layer in enumerate(self.decoding_conv):

            decTensor = decoder_feature_maps[i]
            x = layer(x)
            if self.ghost:
                x = self.merge_concat(decTensor, x)
            else:
                x = ME.cat(decTensor, x)
            x = self.decoding_block[i](x)
            #print("[kpscorenet] layer=",x.F.shape)

        # Note that we skipped ghost masking for the final sparse tensor,
        # namely the tensor with the same resolution as the input to uresnet.
        # This is done at the full chain cnn stage, for consistency with SCN

        device = x.F.device

        x = self.final_block(x)
        score_pred = self.kpscore_pred(x)
        
        print("[kpscorenet] layer=",score_pred.F.shape)

        res = {
            'ppn_kpscore': [score_pred]
        }

        return res


class KeypointScoreNetLoss(torch.nn.modules.loss._Loss):
    """
    Loss function for KeypointScoreNet

    Output
    ------
    reg_loss: float
        Distance loss
    mask_loss: float
        Binary voxel-wise prediction (is there an object of interest or not)
    type_loss: float
        Semantic prediction loss.
    classify_endpoints_loss: float
    classify_endpoints_acc: float

    See Also
    --------
    PPN, mlreco.models.uresnet_ppn_chain
    """

    def __init__(self, cfg, name='kpscorenet'):
        super(KeypointScoreNetLoss, self).__init__()
        self.loss_config = cfg.get('ppn').get('kpscorenet_loss_cfg',{})
        if len(self.loss_config)==0:
            print("missing KeypointScoreNetLoss configuration. looking for key='kpscorenet_loss_cfg' inside 'ppn' config block")
            raise ValueError("Missing configuration")
        self.fn_mse = torch.nn.MSELoss( reduction='none' )
        self.verbose = self.loss_config.get('verbose',True)        
        print("KPScoreNetLoss configured.")

    def forward(self, result, segment_labels, keypoint_labels):

        if self.verbose:
            print("keypointscorenet loss")
            print(" input, result: ",result.keys())
            print(" ppn_kpscore: shape=",result["ppn_kpscore"][0].F.shape)
            print(" segment_labels: ntensors=",len(keypoint_labels))        
            print(" keypoint_labels: ntensors=",len(keypoint_labels))
            print(" keypoint_labels: shape=",keypoint_labels[0].shape)

        pred  = result["ppn_kpscore"][0].F  # score for 6 classes
        label = keypoint_labels[0][:,4:10]  # score for 6 classes. remove (batch,x,y,z)

        # to get a per-voxel weight, we need to get a positive and negative example mask
        with torch.no_grad():
            pos_mask = label>0.05
            neg_mask = label<0.05
            pos_samples = (pos_mask==True).sum()
            neg_samples = (neg_mask==True).sum()

        loss_pervoxel = self.fn_mse( pred, label )

        res = {}
        tot_loss = 0.0

        for kpclass in range(KeypointScoreNet.NUM_KP_CLASSES):            
            # get pos example loss
            with torch.no_grad():
                class_pos_mask = pos_mask[:,kpclass]
                class_neg_mask = neg_mask[:,kpclass]
                class_npos = (class_pos_mask==True).sum()
                class_nneg = (class_neg_mask==True).sum()
            
            loss_pos = loss_pervoxel[:,kpclass][ class_pos_mask ]
            loss_neg = loss_pervoxel[:,kpclass][ class_neg_mask ]
            if self.verbose:
                print("KPCLASS[",kpclass,",",KeypointScoreNet.KP_CLASS_NAMES[kpclass],"] pos loss shape: ",loss_pos.shape)
                print("KPCLASS[",kpclass,",",KeypointScoreNet.KP_CLASS_NAMES[kpclass],"] neg loss shape: ",loss_neg.shape)

            weight_pos = 1.0
            weight_neg = 1.0
            if class_npos>0:
                weight_pos = 1.0/class_npos
            if class_nneg>0:
                weight_neg = 1.0/class_nneg
            loss_class = loss_pos.sum()*weight_pos + loss_neg.sum()*weight_neg
            tot_loss += loss_class/float(KeypointScoreNet.NUM_KP_CLASSES)

        # calculate accuracies
        with torch.no_grad():
            acc = self.keypoint_accuracies( pred.detach(), label.detach(), verbose=self.verbose )
        
        res['loss'] = tot_loss
        res.update(acc)

        if self.verbose:
            with torch.no_grad():
                print("KeypointScoreNet Loss ----------------")
                for x in res:
                    print("  ",x,": ",res[x])
                print("--------------------------------------")
            
        return res

    def keypoint_accuracies( self, kp_pred, kp_label, verbose=False ):
        """
        inputs
        kp_pred: tensor 
          Expects shape (N,class)
        kp_label: tensor
          Expects shape (N,class)

        outputs
        -------

        dict with accuracies for positive and negative examples for each class type
        """
        acc = {}
        tot_corr = 0.0
        tot_calc = 0.0
        for c,kpname in enumerate(KeypointScoreNet.KP_CLASS_NAMES):
            kp_n_pos    = float(kp_label[:,c].ge(0.5).sum().item())
            kp_corr_pos = float(kp_pred[:,c].ge(0.5)[ kp_label[:,c].ge(0.5) ].sum().item())
            if verbose: print("kp[",c,"-",kpname,"] n_pos[>0.5]: ",kp_n_pos," pred[>0.5]: ",kp_corr_pos)
            kp_n_neg    = float(kp_label[:,c].lt(0.5).sum().item())
            kp_corr_neg = float(kp_pred[:,c].lt(0.5)[ kp_label[:,c].lt(0.5) ].sum().item())
            if verbose: print("kp[",c,"-",kpname,"] n_pos[<0.5]: ",kp_n_neg," pred[<0.5]: ",kp_corr_neg)
            if kp_n_pos>0:
                acc[kpname+"_pos"] = kp_corr_pos/kp_n_pos
            else:
                acc[kpname+"_pos"] = 0.0 # None
            if kp_n_neg>0:
                acc[kpname+"_neg"] = kp_corr_neg/kp_n_neg
            else:
                acc[kpname+"_neg"] = 0.0 # None

            tot_corr += kp_corr_neg + kp_corr_pos
            tot_calc += kp_n_pos + kp_n_neg
            
        if tot_calc>0.0:
            acc['accuracy'] = tot_corr/tot_calc
        else:
            acc['accuracy'] = 0.0 # None
            
        return acc
