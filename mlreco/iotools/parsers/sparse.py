import numpy as np
from larcv import larcv

def parse_sparse2d(sparse_event_list):
    """
    A function to retrieve sparse tensor input from larcv::EventSparseTensor2D object

    Returns the data in format to pass to SCN

    .. code-block:: yaml

        schema:
          input_data:
            parser: parse_sparse2d
            args:
              sparse_event_list:
                - sparse2d_pcluster_0 (, 0)
                - sparse2d_pcluster_1 (, 1)
                - ...

    Configuration
    -------------
    sparse_event_list: list of larcv::EventSparseTensor2D
        Optionally, give an array of (larcv::EventSparseTensor2D, int) for projection id

    Returns
    -------
    voxels: np.ndarray(int32)
        Coordinates with shape (N,2)
    data: np.ndarray(float32)
        Pixel values/channels with shape (N,C)
    """
    meta = None
    output = []
    np_voxels = None
    for sparse_event in sparse_event_list:
        projection_id = 0  # default
        if isinstance(sparse_event, tuple):
            projection_id = sparse_event[1]
            sparse_event = sparse_event[0]

        tensor = sparse_event.sparse_tensor_2d(projection_id)
        num_point = tensor.as_vector().size()

        if meta is None:

            meta = tensor.meta()
            np_voxels = np.empty(shape=(num_point, 2), dtype=np.int32)
            larcv.fill_2d_voxels(tensor, np_voxels)

        # else:
        #     assert meta == tensor.meta()
        np_data = np.empty(shape=(num_point, 1), dtype=np.float32)
        larcv.fill_2d_pcloud(tensor, np_data)
        output.append(np_data)
    return np_voxels, np.concatenate(output, axis=-1)


def parse_sparse3d(sparse_event_list, features=None, hit_keys=[], nhits_idx=None):
    """
    A function to retrieve sparse tensor input from larcv::EventSparseTensor3D object

    Returns the data in format to pass to DataLoader

    .. code-block:: yaml

        schema:
          input_data:
            parser: parse_sparse3d
            args:
              sparse_event_list:
                - sparse3d_pcluster_0
                - sparse3d_pcluster_1
                - ...

    Configuration
    -------------
    sparse_event_list: list of larcv::EventSparseTensor3D
        Can be repeated to load more features (one per feature).
    features: int, optional
        Default is None (ignored). If a positive integer is specified,
        the sparse_event_list will be split in equal lists of length
        `features`. Each list will be concatenated along the feature
        dimension separately. Then all lists are concatenated along the
        first dimension (voxels). For example, this lets you work with
        distinct detector volumes whose input data is stored in separate
        TTrees.`features` is required to be a divider of the `sparse_event_list`
        length.
    hit_keys: list of int, optional
        Indices among the input features of the _hit_key_ TTrees that can be
        used to infer the _nhits_ quantity (doublet vs triplet point).
    nhits_idx: int, optional
        Index among the input features where the _nhits_ feature (doublet vs triplet)
        should be inserted.

    Returns
    -------
    voxels: numpy array(int32) with shape (N,3)
        Coordinates
    data: numpy array(float32) with shape (N,C)
        Pixel values/channels, as many channels as specified larcv::EventSparseTensor3D.
    """
    split_sparse_event_list = [sparse_event_list]
    if features is not None and features > 0:
        if len(sparse_event_list) % features > 0:
            raise Exception("features number in parse_sparse3d should be a divider of the sparse_event_list length.")
        split_sparse_event_list = np.split(np.array(sparse_event_list), len(sparse_event_list) / features)

    voxels, features = [], []
    features_count = None
    compute_nhits = len(hit_keys) > 0
    if compute_nhits and nhits_idx is None:
        raise Exception("nhits_idx needs to be specified if you want to compute the _nhits_ feature.")

    for sparse_event_list in split_sparse_event_list:
        if features_count is None:
            features_count = len(sparse_event_list)
        assert len(sparse_event_list) == features_count

        meta = None
        output = []
        np_voxels = None
        hit_key_array = []
        for idx, sparse_event in enumerate(sparse_event_list):
            num_point = sparse_event.as_vector().size()
            if meta is None:
                meta = sparse_event.meta()
                np_voxels = np.empty(shape=(num_point, 3), dtype=np.int32)
                larcv.fill_3d_voxels(sparse_event, np_voxels)
            else:
                assert meta == sparse_event.meta()
            np_data = np.empty(shape=(num_point, 1), dtype=np.float32)
            larcv.fill_3d_pcloud(sparse_event, np_data)
            output.append(np_data)

            if compute_nhits:
                if idx in hit_keys:
                    hit_key_array.append(np_data)

        voxels.append(np_voxels)
        features_array = np.concatenate(output, axis=-1)

        if compute_nhits:
            hit_key_array = np.concatenate(hit_key_array, axis=-1)
            doublets = (hit_key_array < 0).any(axis=1)
            nhits = 3. * np.ones((np_voxels.shape[0],), dtype=np.float32)
            nhits[doublets] = 2.
            if nhits_idx < 0 or nhits_idx > features_array.shape[1]:
                raise Exception("nhits_idx is out of range")
            features_array = np.concatenate([features_array[..., :nhits_idx], nhits[:, None], features_array[..., nhits_idx:]], axis=-1)

        features.append(features_array)

    return np.concatenate(voxels, axis=0), np.concatenate(features, axis=0)


def parse_sparse3d_ghost(sparse_event_semantics):
    """
    A function to retrieve sparse tensor input from larcv::EventSparseTensor3D object

    Converts the sematic class to a ghost vs non-ghost label.

    .. code-block:: yaml

        schema:
          ghost_label:
            parser: parse_sparse3d
            args:
              sparse_event_semantics: sparse3d_semantics

    Configuration
    -------------
    sparse_event_semantics: larcv::EventSparseTensor3D

    Returns
    -------
    np.ndarray
        a numpy array with the shape (N,3+1) where 3+1 represents
        (x,y,z) coordinate and 1 stored ghost labels (channels).
    """
    np_voxels, np_data = parse_sparse3d([sparse_event_semantics])
    return np_voxels, (np_data==5).astype(np.float32)


def parse_sparse3d_charge_rescaled(sparse_event_list):
    # Produces sparse3d_reco_rescaled on the fly on datasets that do not have it
    np_voxels, output = parse_sparse3d(sparse_event_list)

    deghost      = output[:, -1] < 5
    hit_charges  = output[deghost,  :3]
    hit_ids      = output[deghost, 3:6]
    pmask        = hit_ids > -1

    _, inverse, counts = np.unique(hit_ids, return_inverse=True, return_counts=True)
    multiplicity = counts[inverse].reshape(-1,3)
    charges = np.sum((hit_charges*pmask)/multiplicity, axis=1)/np.sum(pmask, axis=1)

    return np_voxels[deghost], charges.reshape(-1,1)

__DROPME__ = None
def parse_sparse3d_drop_cosmics(sparse_event_list, sparse_origin_index=-1, drop_cosmic_prob=0.5 ):
    """
    A function to retrieve sparse tensor input from larcv::EventSparseTensor3D object
    Drop cosmics using at random.
    This is used to help focus training on neutrino voxels which 
      occur at lower rates than cosmics and do look quite different.

    Returns the data in format to pass to DataLoader

    .. code-block:: yaml

        schema:
          input_data:
            parser: parse_sparse3d
            args:
              sparse_event_list:
                - sparse3d_pcluster_0
                - sparse3d_pcluster_1
                - ...
              sparse_origin_index: -1
              drop_cosmic_prob: float or None
               

    Configuration
    -------------
    sparse_event_list: list of larcv::EventSparseTensor3D
        Can be repeated to load more features (one per feature).
    sparse_origin_index: index in sparse_event_list that should be used as the origin feature.
        Origin flags not included in output faeture tensor.
    drop_cosmic_prob: for voxels tagged as cosmic, probability all will be dropped from the event

    Returns
    -------
    voxels: numpy array(int32) with shape (N,3)
        Coordinates
    data: numpy array(float32) with shape (N,C)
        Pixel values/channels, as many channels as specified larcv::EventSparseTensor3D.
    """
    global __DROPME__
    np_voxels = None
    features  = []
    origin_index = sparse_origin_index
    if origin_index<0:
        origin_index = len(sparse_event_list)+origin_index
    print("num sparse tensors: ",len(sparse_event_list))
    print("origin index: ",origin_index)

    if type(drop_cosmic_prob) is str:
        if drop_cosmic_prob=="None":
            drop_cosmic_prob = None
        else:
            drop_cosmic_prob = float(drop_cosmic_prob)
        print("drop_cosmic_prob: ",drop_cosmic_prob," ",type(drop_cosmic_prob))

    meta = None
    num_point = None
    for idx, sparse_event in enumerate(sparse_event_list):
        
        if idx==origin_index:
            continue
        
        if meta is None:
            num_point = sparse_event.as_vector().size()            
            meta = sparse_event.meta()
            np_voxels = np.empty(shape=(num_point, 3), dtype=np.int32)
            larcv.fill_3d_voxels(sparse_event, np_voxels)
        else:
            assert meta == sparse_event.meta()
        np_data = np.empty(shape=(num_point, 1), dtype=np.float32)
        larcv.fill_3d_pcloud(sparse_event, np_data)
        features.append(np_data)
        
    np_features = np.concatenate(features, axis=-1)

    np_origin = np.empty(shape=(num_point,1),dtype=np.float32)
    larcv.fill_3d_pcloud(sparse_event_list[origin_index],np_origin)

    non_cosmic_mask = (np_origin[:,0]).astype(np.int)!=1
    print("np_voxels: ",np_voxels.shape)    
    print("np_data: ",np_data.shape)
    print("np_origin: ",np_origin.shape)    
    print("mask shape: ",non_cosmic_mask.shape)
    print("nu voxels: ",non_cosmic_mask.sum())

    if drop_cosmic_prob is not None:
        if np.random.uniform()<drop_cosmic_prob:
            __DROPME__ = True
        else:
            __DROPME__ = False
            
        print("rolled the dice to drop cosmics: ",__DROPME__)

        if __DROPME__:
            return np_voxels[non_cosmic_mask[:],:],np_features[non_cosmic_mask[:],:]
        else:
            return np_voxels,np_features

    if drop_cosmic_prob is None and __DROPME__==True:
        print("follow previous rolls to drop cosmics: ",__DROPME__)        
        return np_voxels[non_cosmic_mask[:],:],np_features[non_cosmic_mask[:],:]

    return np_voxels,np_features
