
# SETUP ENVIRONMENT VARIABLES FOR MLRECO
import os,sys
SOFTWARE_DIR = '/home/twongjirad/working/larbys/lartpc_mlreco3d/'
DATA_DIR = os.environ.get('DATA_DIR')

# IMPORTS
import numpy as np
import yaml
import torch
import ROOT as rt
from array import array

from mlreco.utils.ppn import uresnet_ppn_type_point_selector
from mlreco.utils.cluster.dense_cluster import fit_predict_np, gaussian_kernel
from mlreco.main_funcs import process_config, prepare
from mlreco.utils.gnn.cluster import get_cluster_label
from mlreco.utils.deghosting import adapt_labels_numpy as adapt_labels

from larcv import larcv

# LOAD THE YAML CONFIG FILE
cfg=yaml.load(open('%s/config/train_ubmlreco_uresnet_ppn.cfg' % (SOFTWARE_DIR), 'r').read(),Loader=yaml.Loader)

# OUR CHANCE TO MODIFY THE DEFAULTS
cfg['trainval']['train'] = False
#cfg['trainval']['model_path'] = "location of model weights"
#print(cfg['iotool']['dataset']['data_keys']) # list of files to load
#cfg['iotool']['dataset']['data_keys'] = ["file1.root","file2.root"]
#cfg['iotool']['batchsize'] = 1

# DUMP OUT THE YAML
#print(yaml.dump(cfg, default_flow_style=False))

# let mlreco process the configuration
process_config(cfg)

# get handlers for data, model
hs=prepare(cfg)

NENTRIES = len(hs.data_io)
print("NENTRIES: ",NENTRIES)

# OUTPUT ROOT FILE
out = rt.TFile("output.root","recreate")

# create a ROOT tree that will save data
outtree = rt.TTree("analysis","My analysis tree")

# define some branch variables that we store into the ROOT tree
an_int   = array('i',[0]) # an example integer
an_float = array('f',[0.0]) # an example float
an_float_array = array('f',[0.0,0.0]) # an example float array

# register the variables to the branches
outtree.Branch("x",an_int,'x/I')
outtree.Branch("y",an_float,'y/F')
outtree.Branch("z",an_float_array,'z[2]/F')


for ientry in range(NENTRIES):
    # get next entry using data_io_iter and then pass it through the network chain
    data, output = hs.trainer.forward(hs.data_io_iter)

    batchidx = 0 # batchsize 1, so only 1 batch index

    # the proposed voxels we passed into the net, based on the wire plane images only
    input_data = data['input_data'][batchidx]
    print("input_data: ",input_data.shape)

    # the true particle ID labels from MC truth
    segment_label = data['segment_label'][batchidx][:, -1]
    print("segment_label: ",segment_label.shape)

    # the voxels the network thinks are ghost
    ghost_mask = output['ghost'][batchidx].argmax(axis=1) == 0
    print("ghost_mask: ",ghost_mask.shape)

    # the predicted particle class produced by the network
    segment_pred = output['segmentation'][batchidx].argmax(axis=1)
    print("segment_pred: ",segment_pred.shape)

    # fill branch variables with stuff
    an_int[0] = ientry+1
    an_float[0] = (ientry+1)*2.0
    an_float_array[0] = (ientry+1)*3.0
    an_float_array[1] = (ientry+1)*4.0

    outtree.Fill()
    break

outtree.Write()
out.Close()

outmsg = """
You can check the contents of your ROOT file by:

root -l output.root
.ls
analysis->Scan()

You'll see something like the following:

twongjirad@pop-os:~/working/larbys/lartpc_mlreco3d$ root -l output.root 
   ------------------------------------------------------------------
  | Welcome to ROOT 6.24/04                        https://root.cern |
  | (c) 1995-2021, The ROOT Team; conception: R. Brun, F. Rademakers |
  | Built for linuxx8664gcc on Dec 28 2021, 13:13:00                 |
  | From tag , 25 August 2021                                        |
  | With                                                             |
  | Try '.help', '.demo', '.license', '.credits', '.quit'/'.q'       |
   ------------------------------------------------------------------

root [0] 
Attaching file output.root as _file0...
(TFile *) 0x55f30b36df80
root [1] .ls
TFile**		output.root	
 TFile*		output.root	
  KEY: TTree	analysis;1	My analysis tree
root [2] analysis->Scan()
***********************************************************
*    Row   * Instance *       x.x *       y.y *       z.z *
***********************************************************
*        0 *        0 *         1 *         2 *         3 *
*        0 *        1 *         1 *         2 *         4 *
***********************************************************
(long long) 2


"""
print(outmsg)


