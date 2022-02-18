import os,sys
import mlreco.iotools.parser_factory

def parse_ub_sparse3d_me(data):
    """
    A function to retrieve sparse tensor input from larcv::NumpyArrayInt/Float objects

    Returns the data in format to pass to MinkowskiEngine

    Parameters
    ----------
    data: list
        Array of vector<larcv::NumpyArrayInt/Float>

    Returns
    -------
    voxels: numpy array(int32) with shape (N,3)
        Coordinates
    data: numpy array(float32) with shape (N,C)
        Pixel values/channels
    """
    coord_np = data[0][0].at(0).tonumpy()
    feat_np  = data[1][0].at(0).tonumpy()
    if len(feat_np.shape)==1:
        feat_np = feat_np.reshape( feat_np.shape[0], 1 )
    return coord_np,feat_np
mlreco.iotools.parser_factory.register_parser( "parse_ub_sparse3d_me", parse_ub_sparse3d_me )

def parse_ub_particle_points(data,include_point_tagging=False):
    """
    A function to retrieve particles ground truth points tensor, returns
    points coordinates, types and particle index.

    Parameters
    ----------
    data: list
        length 2 array of larcv::EventSparseTensor3D and larcv::EventParticle

    Returns
    -------
    np_voxels: np.ndarray
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
    np_values: np.ndarray
        a numpy array with the shape (N, 2) where 2 represents the class of the ground truth point
        and the particle data index in this order. (optionally: end/start tagging)
    """
    kp_vox_pos = data[0][0].at(0).tonumpy()
    kp_labels  = data[1][0].at(0).tonumpy()
    return kp_vox_pos,kp_labels
mlreco.iotools.parser_factory.register_parser( "parse_ub_particle_points", parse_ub_particle_points )

