import random
import torch
import numpy as np

from mlreco.models.layers.common.dbscan import DBSCANFragmenter
from mlreco.models.layers.common.momentum import DeepVertexNet, EvidentialMomentumNet, MomentumNet, VertexNet
from mlreco.models.experimental.transformers.transformer import TransformerEncoderLayer
from mlreco.models.layers.gnn import gnn_model_construct, node_encoder_construct, edge_encoder_construct, node_loss_construct, edge_loss_construct

from mlreco.utils.gnn.data import merge_batch, split_clusts, split_edge_index
from mlreco.utils.gnn.cluster import form_clusters, get_cluster_batch, get_cluster_label, get_cluster_points_label, get_cluster_directions, get_cluster_dedxs
from mlreco.utils.gnn.network import complete_graph, delaunay_graph, mst_graph, bipartite_graph, inter_cluster_distance, knn_graph

class GNN(torch.nn.Module):
    """
    Driver class for cluster node+edge prediction, assumed to be a GNN model.

    This class mostly acts as a wrapper that will hand the graph data to another model.
    If DBSCAN is used, use the semantic label tensor as an input.

    Typical configuration can look like this:

    .. code-block:: yaml

        model:
          name: grappa
          modules:
            grappa:
              your config goes here

    Configuration
    -------------
    base: dict
        Configuration of base Grappa :

        .. code-block:: yaml

          base:
            node_type         : <semantic class to group (all classes if -1, default 0, i.e. EM)>
            node_min_size     : <minimum number of voxels inside a cluster to be considered (default -1)>
            source_col        : <column in the input data that specifies the source node ids of each voxel (default 5)>
            target_col        : <column in the input data that specifies the target instance ids of each voxel (default 6)>
            use_dbscan        : <use DBSCAN to cluster the input instances of the class specified by node_type (default False)>
            add_start_point   : <add label start point to the node features (default False)>
            add_start_dir     : <add predicted start direction to the node features (default False)>
            start_dir_max_dist: <maximium distance between start point and cluster voxels to be used to estimate direction (default -1, i.e no limit)>
            start_dir_opt     : <optimize start direction by minimizing relative transverse spread of neighborhood (slow, default: False)>
            add_start_dedx    : <add predicted start dedx to the node features (default False)>
            network           : <type of network: 'complete', 'delaunay', 'mst', 'knn' or 'bipartite' (default 'complete')>
            edge_max_dist     : <maximal edge Euclidean length (default -1)>
            edge_dist_method  : <edge length evaluation method: 'centroid' or 'set' (default 'set')>
            merge_batch       : <flag for whether to merge batches (default False)>
            merge_batch_mode  : <mode of batch merging, 'const' or 'fluc'; 'const' use a fixed size of batch for merging, 'fluc' takes the input size a mean and sample based on it (default 'const')>
            merge_batch_size  : <size of batch merging (default 2)>
            shuffle_clusters  : <randomize cluster order (default False)>
            kinematics_mlp    : <applies type and momentum MLPs on the node features (default False)>

    dbscan: dict
        dictionary of dbscan parameters
    node_encoder: dict

        .. code-block:: yaml

          node_encoder:
            name: <name of the node encoder>
            <dictionary of arguments to pass to the encoder>
            model_path      : <path to the encoder weights>

    edge_encoder: dict

        .. code-block:: yaml

          edge_encoder:
            name: <name of the edge encoder>
            <dictionary of arguments to pass to the encoder>
            model_path      : <path to the encoder weights>

    gnn_model: dict
        .. code-block:: yaml

          gnn_model:
            name: <name of the node model>
            <dictionary of arguments to pass to the model>
            model_path      : <path to the model weights>

    kinematics_mlp: bool, default False
        Whether to enable MLP-like layers after the GNN to predict
        momentum, particle type, etc.
    kinematics_type: bool
        Whether to add PID MLP to each node.
    kinematics_momentum: bool
        Whether to add momentum MLP to each node.
    type_net: dict
        Configuration for the PID MLP (if enabled).
        Can partial load weights here too.
    momentum_net: dict
        Configuration for the Momentum MLP (if enabled).
        Can partial load weights here too.
    vertex_mlp: bool, default False
        Whether to add vertex prediction MLP to each node.
        Includes primary particle + vertex coordinates predictions.
    vertex_net: dict
        Configuration for the Vertex MLP (if enabled).
        Can partial load weights here too.

    Outputs
    -------
    input_node_features:
    input_edge_features:
    clusts:
    edge_index:
    node_pred:
    edge_pred:
    node_pred_p:
    node_pred_type:
    node_pred_vtx:

    See Also
    --------
    GNNLoss
    """

    MODULES = [('grappa', ['base', 'dbscan', 'node_encoder', 'edge_encoder', 'gnn_model']), 'grappa_loss']

    def __init__(self, cfg, name='grappa', batch_col=0, coords_col=(1, 4)):
        super(GNN, self).__init__()

        # Get the chain input parameters
        base_config = cfg[name].get('base', {})
        self.name = name

        # Choose what type of node to use
        self.node_type = base_config.get('node_type', 0)
        self.node_min_size = base_config.get('node_min_size', -1)
        self.source_col = base_config.get('source_col', 5)
        self.target_col = base_config.get('target_col', 6)
        self.add_start_point = base_config.get('add_start_point', False)
        self.add_start_dir = base_config.get('add_start_dir', False)
        self.start_dir_max_dist = base_config.get('start_dir_max_dist', -1)
        self.start_dir_opt = base_config.get('start_dir_opt', False)
        self.add_start_dedx = base_config.get('add_start_dedx', False)
        self.shuffle_clusters = base_config.get('shuffle_clusters', False)

        self.batch_index = batch_col
        self.coords_index = coords_col

        # Interpret node type as list of classes to cluster, -1 means all classes
        if isinstance(self.node_type, int): self.node_type = [self.node_type]

        # Choose what type of network to use
        self.network = base_config.get('network', 'complete')
        self.edge_max_dist = base_config.get('edge_max_dist', -1)
        self.edge_dist_metric = base_config.get('edge_dist_metric', 'set')
        self.edge_knn_k = base_config.get('edge_knn_k', 5)

        # If requested, merge images together within the batch
        self.merge_batch = base_config.get('merge_batch', False)
        self.merge_batch_mode = base_config.get('merge_batch_mode', 'const')
        self.merge_batch_size = base_config.get('merge_batch_size', 2)
        if self.merge_batch_mode not in ['const', 'fluc']:
            raise ValueError('Batch merging mode not supported, must be one of const or fluc')
        self.merge_batch_fluc = self.merge_batch_mode == 'fluc'

        # If requested, use DBSCAN to form clusters from semantics
        if 'dbscan' in cfg[name]:
            cfg[name]['dbscan']['cluster_classes'] = self.node_type if self.node_type[0] > -1 else [0,1,2,3]
            cfg[name]['dbscan']['min_size']        = self.node_min_size
            self.dbscan = DBSCANFragmenter(cfg[name], name='dbscan',
                                            batch_col=self.batch_index,
                                            coords_col=self.coords_index)

        # If requested, initialize two MLPs for kinematics predictions
        self.kinematics_mlp = base_config.get('kinematics_mlp', False)
        self.kinematics_type = base_config.get('kinematics_type', False)
        self.kinematics_momentum = base_config.get('kinematics_momentum', False)
        if self.kinematics_mlp:
            node_output_feats = cfg[name]['gnn_model'].get('node_output_feats', 64)
            self.kinematics_type = base_config.get('kinematics_type', False)
            self.kinematics_momentum = base_config.get('kinematics_momentum', False)
            if self.kinematics_type:
                type_config = cfg[name].get('type_net', {})
                type_net_mode = type_config.get('mode', 'standard')
                if type_net_mode == 'standard':
                    self.type_net = MomentumNet(node_output_feats,
                                                num_output=5,
                                                num_hidden=type_config.get('num_hidden', 128),
                                                positive_outputs=False)
                elif type_net_mode == 'edl':
                    self.type_net = MomentumNet(node_output_feats,
                                                num_output=5,
                                                num_hidden=type_config.get('num_hidden', 128),
                                                positive_outputs=True)
                    self.edge_softplus = torch.nn.Softplus()
                else:
                    raise ValueError('Unrecognized Particle ID Type Net Mode: ', type_net_mode)
            if self.kinematics_momentum:
                momentum_config = cfg[name].get('momentum_net', {})
                softplus_and_shift = momentum_config.get('eps', 0.0)
                logspace = momentum_config.get('logspace', False)
                if momentum_config.get('mode', 'standard') == 'edl':
                    self.momentum_net = EvidentialMomentumNet(node_output_feats,
                                                              num_output=4,
                                                              num_hidden=momentum_config.get('num_hidden', 128),
                                                              eps=softplus_and_shift,
                                                              logspace=logspace)
                else:
                    self.momentum_net = MomentumNet(node_output_feats,
                                                    num_output=1,
                                                    num_hidden=momentum_config.get('num_hidden', 128))

        self.vertex_mlp = base_config.get('vertex_mlp', False)
        if self.vertex_mlp:
            node_output_feats = cfg[name]['gnn_model'].get('node_output_feats', 64)
            vertex_config = cfg[name].get('vertex_net', {'name': 'momentum_net'})
            self.use_vtx_input_features = vertex_config.get('use_vtx_input_features', False)
            vertex_net_name = vertex_config.get('name', 'momentum_net')
            if vertex_net_name == 'momentum_net':
                self.vertex_net = VertexNet(node_output_feats,
                                              num_output=5,
                                              num_hidden=vertex_config.get('num_hidden', 64)) # Enforce positive outputs
            elif vertex_net_name == 'attention_net':
                self.vertex_net = TransformerEncoderLayer(node_output_feats, 3, **vertex_config)
            elif vertex_net_name == 'deep_vertex_net':
                self.vertex_net = DeepVertexNet(node_output_feats,
                                                  num_output=5,
                                                  num_hidden=vertex_config.get('num_hidden', 64),
                                                  num_layers=vertex_config.get('num_layers', 5)) # Enforce positive outputs
            else:
                raise ValueError('Vertex MLP {} not recognized!'.format(vertex_config['name']))

        # Initialize encoders
        self.node_encoder = node_encoder_construct(cfg[name], batch_col=self.batch_index, coords_col=self.coords_index)
        self.edge_encoder = edge_encoder_construct(cfg[name], batch_col=self.batch_index, coords_col=self.coords_index)

        # Construct the GNN
        self.gnn_model = gnn_model_construct(cfg[name])

    def forward(self, data, clusts=None, groups=None, points=None, extra_feats=None, batch_size=None):
        """
        Prepares particle clusters and feed them to the GNN model.

        Args:
            array:
                data[0] ([torch.tensor]): (N,5-10) [x, y, z, batch_id(, value), part_id(, group_id, int_id, nu_id, sem_type)]
                                       or (N,5) [x, y, z, batch_id, sem_type] (with DBSCAN)
                data[1] ([torch.tensor]): (N,8) [first_x, first_y, first_z, batch_id, last_x, last_y, last_z, first_step_t] (optional)
            clusts: [(N_0), (N_1), ..., (N_C)] Cluster ids (optional)
            groups: (C) vectors of groups IDs (one per cluster) to enforce connections only within each group
            points: (N,3/6) tensor of start (and end) points of clusters
            extra_feats: (N,F) tensor of features to add to the encoded features
        Returns:
            dict:
                'node_pred' (torch.tensor): (N,2) Two-channel node predictions (split batch-wise)
                'edge_pred' (torch.tensor): (E,2) Two-channel edge predictions (split batch-wise)
                'clusts' ([np.ndarray])   : [(N_0), (N_1), ..., (N_C)] Cluster ids (split batch-wise)
                'edge_index' (np.ndarray) : (E,2) Incidence matrix (split batch-wise)
        """
        cluster_data = data[0]
        if len(data) > 1: particles = data[1]
        result = {}

        # Form list of list of voxel indices, one list per cluster in the requested class
        if clusts is None:
            if hasattr(self, 'dbscan'):
                clusts = self.dbscan(cluster_data, points=particles if len(data) > 1 else None)
            else:
                clusts = form_clusters(cluster_data.detach().cpu().numpy(),
                                       self.node_min_size,
                                       self.source_col,
                                       cluster_classes=self.node_type)

        # If requested, shuffle the order in which the clusters are listed (used for debugging)
        if self.shuffle_clusters:
            random.shuffle(clusts)

        # If requested, merge images together within the batch
        if self.merge_batch:
            cluster_data, particles, batch_list = merge_batch(cluster_data, particles, self.merge_batch_size, self.merge_batch_fluc, self.batch_index)
            batch_counts = np.unique(batch_list, return_counts=True)[1]
            result['batch_counts'] = [batch_counts]

        # Update result with a list of clusters for each batch id
        batches, bcounts = np.unique(cluster_data[:,self.batch_index].detach().cpu().numpy(), return_counts=True)
        if not len(clusts):
            return {**result, 'clusts': [[np.array([]) for _ in batches]]}

        # If an event is missing from the input data - e.g., deghosting
        # erased everything (extreme case but possible if very few voxels)
        # then we might be miscounting batches. Ensure that batches is the
        # same length as batch_size if specified.
        if batch_size is not None:
            new_bcounts = np.zeros(batch_size, dtype=np.int64)
            new_bcounts[batches.astype(np.int64)] = bcounts
            bcounts = new_bcounts
            batches = np.arange(batch_size)

        batch_ids = get_cluster_batch(cluster_data, clusts, batch_index=self.batch_index)
        clusts_split, cbids = split_clusts(clusts, batch_ids, batches, bcounts)
        result['clusts'] = [clusts_split]

        # If necessary, compute the cluster distance matrix
        dist_mat = None
        if self.edge_max_dist > 0 or self.network == 'mst' or self.network == 'knn':
            dist_mat = inter_cluster_distance(cluster_data[:,self.coords_index[0]:self.coords_index[1]], clusts, batch_ids, self.edge_dist_metric)

        # Form the requested network
        if len(clusts) == 1:
            edge_index = np.empty((2,0), dtype=np.int64)
        elif self.network == 'complete':
            edge_index = complete_graph(batch_ids, dist_mat, self.edge_max_dist)
        elif self.network == 'delaunay':
            import numba as nb
            edge_index = delaunay_graph(cluster_data.cpu().numpy(), nb.typed.List(clusts), batch_ids, dist_mat, self.edge_max_dist,
                                        batch_col=self.batch_index, coords_col=self.coords_index)
        elif self.network == 'mst':
            edge_index = mst_graph(batch_ids, dist_mat, self.edge_max_dist)
        elif self.network == 'knn':
            edge_index = knn_graph(batch_ids, self.edge_knn_k, dist_mat)
        elif self.network == 'bipartite':
            clust_ids = get_cluster_label(cluster_data, clusts, self.source_col)
            group_ids = get_cluster_label(cluster_data, clusts, self.target_col)
            edge_index = bipartite_graph(batch_ids, clust_ids==group_ids, dist_mat, self.edge_max_dist)
        else:
            raise ValueError('Network type not recognized: '+self.network)

        # If groups is sepecified, only keep edges that belong to the same group (cluster graph)
        if groups is not None:
            mask = groups[edge_index[0]] == groups[edge_index[1]]
            edge_index = edge_index[:,mask]

        # Update result with a list of edges for each batch id
        edge_index_split, ebids = split_edge_index(edge_index, batch_ids, batches)
        result['edge_index'] = [edge_index_split]

        # Obtain node and edge features
        x = self.node_encoder(cluster_data, clusts)
        e = self.edge_encoder(cluster_data, clusts, edge_index)

        # If extra features are provided separately, add them
        if extra_feats is not None:
            x = torch.cat([x, extra_feats.float()], dim=1)

        # Add start point and/or start direction to node features if requested
        if self.add_start_point or points is not None:
            if points is None:
                points = get_cluster_points_label(cluster_data, particles, clusts, coords_index=self.coords_index)
            x = torch.cat([x, points.float()], dim=1)
            if self.add_start_dir:
                dirs = get_cluster_directions(cluster_data[:, self.coords_index[0]:self.coords_index[1]], points[:,:3], clusts, self.start_dir_max_dist, self.start_dir_opt)
                x = torch.cat([x, dirs.float()], dim=1)
            if self.add_start_dedx:
                dedxs = get_cluster_dedxs(cluster_data[:, self.coords_index[0]:self.coords_index[1]], cluster_data[:,4], points[:,:3], clusts, self.start_dir_max_dist)
                x = torch.cat([x, dedxs.reshape(-1,1).float()], dim=1)

        # Bring edge_index and batch_ids to device
        index = torch.tensor(edge_index, device=cluster_data.device, dtype=torch.long)
        xbatch = torch.tensor(batch_ids, device=cluster_data.device)

        result['input_node_features'] = [[x[b] for b in cbids]]
        result['input_edge_features'] = [[e[b] for b in ebids]]

        # Pass through the model, update results
        out = self.gnn_model(x, index, e, xbatch)
        result['node_pred'] = [[out['node_pred'][0][b] for b in cbids]]
        result['edge_pred'] = [[out['edge_pred'][0][b] for b in ebids]]

        # If requested, pass the node features through two MLPs for kinematics predictions
        if self.kinematics_mlp:
            if self.kinematics_type:
                node_pred_type = self.type_net(out['node_features'][0])
                result['node_pred_type'] = [[node_pred_type[b] for b in cbids]]
            if self.kinematics_momentum:
                node_pred_p = self.momentum_net(out['node_features'][0])
                if isinstance(self.momentum_net, EvidentialMomentumNet):
                    result['node_pred_p'] = [[node_pred_p[b] for b in cbids]]
                    aleatoric = node_pred_p[:, 3] / (node_pred_p[:, 2] - 1.0 + 0.001)
                    epistemic = node_pred_p[:, 3] / (node_pred_p[:, 1] * (node_pred_p[:, 2] - 1.0 + 0.001))
                    result['node_pred_p_aleatoric'] = [[aleatoric[b] for b in cbids]]
                    result['node_pred_p_epistemic'] = [[epistemic[b] for b in cbids]]
                else:
                    result['node_pred_p'] = [[node_pred_p[b] for b in cbids]]
        else:
            # If final post-gnn MLP is not given, set type features to node_pred.
            result['node_pred_type'] = result['node_pred']

        if self.vertex_mlp:
            if self.use_vtx_input_features:
                node_pred_vtx = self.vertex_net(x)
            else:
                node_pred_vtx = self.vertex_net(out['node_features'][0])
            result['node_pred_vtx'] = [[node_pred_vtx[b] for b in cbids]]

        return result


class GNNLoss(torch.nn.modules.loss._Loss):
    """
    Takes the output of the GNN and computes the total loss.

    For use in config:

    ..  code-block:: yaml

        model:
          name: grappa
          modules:
            grappa_loss:
              node_loss:
                name: <name of the node loss>
                <dictionary of arguments to pass to the loss>
              edge_loss:
                name: <name of the edge loss>
                <dictionary of arguments to pass to the loss>
    """
    def __init__(self, cfg, name='grappa_loss', batch_col=0, coords_col=(1, 4)):
        super(GNNLoss, self).__init__()

        self.batch_index = batch_col
        self.coords_index = coords_col

        # Initialize the node and edge losses, if requested
        self.apply_node_loss, self.apply_edge_loss = False, False
        if 'node_loss' in cfg[name]:
            self.apply_node_loss = True
            self.node_loss = node_loss_construct(cfg[name], batch_col=batch_col, coords_col=coords_col)
        if 'edge_loss' in cfg[name]:
            self.apply_edge_loss = True
            self.edge_loss = edge_loss_construct(cfg[name], batch_col=batch_col, coords_col=coords_col)


    def forward(self, result, clust_label, graph=None, node_label=None, iteration=None):

        # Apply edge and node losses, if instantiated
        loss = {}
        if self.apply_node_loss:
            if node_label is None:
                node_label = clust_label
            if iteration is not None:
                node_loss = self.node_loss(result, node_label, iteration=iteration)
            else:
                node_loss = self.node_loss(result, node_label)
            loss.update(node_loss)
            loss['node_loss'] = node_loss['loss']
            loss['node_accuracy'] = node_loss['accuracy']
        if self.apply_edge_loss:
            edge_loss = self.edge_loss(result, clust_label, graph)
            loss.update(edge_loss)
            loss['edge_loss'] = edge_loss['loss']
            loss['edge_accuracy'] = edge_loss['accuracy']
        if self.apply_node_loss and self.apply_edge_loss:
            loss['loss'] = loss['node_loss'] + loss['edge_loss']
            loss['accuracy'] = (loss['node_accuracy'] + loss['edge_accuracy'])/2

        return loss
