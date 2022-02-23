import os,sys
import numpy as np
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
    vtxvox     = data[2][0]
    print("pre-filter num keypoints: ",kp_vox_pos.shape[0])
    print("kplabels vtxvox: ",(vtxvox[0],vtxvox[1],vtxvox[2]))
    cropsize = [768,768,768]
    
    xpos = np.copy(kp_vox_pos)
    xlabels = np.copy(kp_labels)

    for i in range(3):
        xfilter = xpos[:,i]>=vtxvox[i]
        xpos = xpos[ xfilter[:], : ]
        xlabels = xlabels[ xfilter[:], : ]

        xfilter = xpos[:,i]<vtxvox[i]+cropsize[i]
        xpos = xpos[ xfilter[:], : ]
        xlabels = xlabels[ xfilter[:], : ]

        xpos[:,i] -= vtxvox[i]
    print("post-filter num keypoints: xpos=",xpos.shape," xfeat=",xlabels.shape)
    
    return xpos,xlabels
mlreco.iotools.parser_factory.register_parser( "parse_ub_particle_points", parse_ub_particle_points )

def parse_ub_cropped_sparse3d_me(data):
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
    print("[parse_ub_cropped_sparse3d_me] ===================================")
    coord_np = data[0][0].at(0).tonumpy()
    feat_np  = data[1][0].at(0).tonumpy()
    kpvox_np = data[2][0].at(0).tonumpy()
    vtxvox   = data[3][0]
    print("ub cropped sparsed3d me, nu vertex voxel: ",(vtxvox[0],vtxvox[1],vtxvox[2]))
    

    voxlimits = [2400+1008*6,117*2,1036]
    voxlimits[0] = (voxlimits[0]-3200.0)*0.5*0.111 # ticks to cm
    for i in range(3):
        voxlimits[i] = int(voxlimits[i]/0.3+0.5) # the max voxel
    print("voxlimits: ",voxlimits)
    cropsize = [768,768,768]

    # choose a keypoint to crop around
    npts = kpvox_np.shape[0]
    attempts = []
    while len(attempts)<npts:
        ikp = np.random.randint( kpvox_np.shape[0] )
        attempts.append(ikp)
        print("choose ikp=",ikp," of ",kpvox_np.shape[0])
        cropvtx = kpvox_np[ikp,:]
        print("seed vtx voxel: ",cropvtx," ",cropvtx.shape)
        jitter = np.random.randint(-50,51,size=3)
        print("jitter: ",jitter," ",jitter.shape)
        proposedvtx = cropvtx+jitter
        # check the bounds
        for i in range(3):
            if proposedvtx[i]-(cropsize[i]/2+1)<0:
                proposedvtx[i] = cropsize[i]/2+1
            if proposedvtx[i]+(cropsize[i]/2)+1>=voxlimits[i]:
                proposedvtx[i] = int(voxlimits[i])-1-int(cropsize[i]/2)
        print("proposedvtx (post-checks): ",proposedvtx)

        # we have to crop coords and feats
        xcoord = np.copy( coord_np )
        xfeat  = np.copy( feat_np )
        print("coord_np: ",coord_np.shape)
        print("feat_np: ",feat_np.shape)
        for i in range(3):
            lowpt = int(proposedvtx[i]-int(cropsize[i]/2))
            print("dim[%d] lowpt=%d"%(i,lowpt))
            xfilter = xcoord[:,i]>=lowpt
            xcoord = xcoord[xfilter[:],:]
            xfeat  = xfeat[xfilter[:],:]
            
            xfilter = xcoord[:,i]<lowpt+cropsize[i]
            xcoord = xcoord[xfilter[:],:]
            xfeat  = xfeat[xfilter[:],:]

            xcoord[:,i] -= lowpt

            print("postfilter dim=",i,": ",xcoord.shape," ",xfeat.shape)

        # check it
        isok = True
        for i in range(3):
            print("dim[",i,"] min-check: ",xcoord[xcoord<0])
            print("dim[",i,"] max-check: ",xcoord[xcoord>=cropsize[i]])
            if np.sum(xcoord<0)>0 or np.sum(xcoord>=cropsize[i])>0:
                isok = False

        if xcoord.shape[0]<1000:
            isok = False
            
        if isok:
            print("good crop")
            break
        else:
            print("bad crop")
            continue
    
    for i in range(3):
        data[3][0][i] = int(proposedvtx[i]-int(cropsize[i]/2))
    if len(xfeat.shape)==1:
        xfeat = xfeat.reshape( feat_np.shape[0], 1 )

    print("========================== end of [parse_ub_cropped_sparse3d_me]")
    return xcoord,xfeat
mlreco.iotools.parser_factory.register_parser( "parse_ub_cropped_sparse3d_me", parse_ub_cropped_sparse3d_me )

def parse_ub_cropped_segment3d_me(data):
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
    print("[parse_ub_cropped_segment3d_me] ===================================")
    coord_np = data[0][0].at(0).tonumpy()
    feat_np  = data[1][0].at(0).tonumpy()
    if len(feat_np.shape)==1:
        feat_np = feat_np.reshape( feat_np.shape[0], 1 )
    vtxvox   = data[2][0]
    print("origin voxel of crop: ",(vtxvox[0],vtxvox[1],vtxvox[2]))
    cropsize = [768,768,768]

    # crop the coords and feats
    xcoord = np.copy( coord_np )
    xfeat  = np.copy( feat_np )
    print("coord_np: ",coord_np.shape)
    print("feat_np: ",feat_np.shape)
    for i in range(3):
        lowpt = int(vtxvox[i])
        print("dim[%d] lowpt=%d"%(i,lowpt))
        xfilter = xcoord[:,i]>=lowpt
        xcoord = xcoord[xfilter[:],:]
        xfeat  = xfeat[xfilter[:]]
            
        xfilter = xcoord[:,i]<lowpt+cropsize[i]
        xcoord = xcoord[xfilter[:],:]
        xfeat  = xfeat[xfilter[:]]

        xcoord[:,i] -= lowpt
        print("postfilter dim=",i,": ",xcoord.shape," ",xfeat.shape)
    print("========================== end of [parse_ub_cropped_segment3d_me]")
    return xcoord,xfeat
mlreco.iotools.parser_factory.register_parser( "parse_ub_cropped_segment3d_me", parse_ub_cropped_segment3d_me )
