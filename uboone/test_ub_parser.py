import os,sys,argparse
import yaml
import numpy as np
sys.path.append("../")
import mlreco
from mlreco.main_funcs import process_config
from mlreco.iotools.factories import loader_factory
import mlreco.iotools.ubparsers

"""
Test of UB Parsers found in mlreco.iotools.parsers


"""
cfg_file = "test_ub_loader.yaml"
if not os.path.isfile(cfg_file):
    cfg_file = os.path.join(current_directory, 'config', config)
if not os.path.isfile(cfg_file):
    print(config, 'not found...')
    sys.exit(1)

cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)

process_config(cfg)

data_io = loader_factory(cfg, event_list=None)

def cycle(data_io):
    while True:
        for x in data_io:
            yield x

data_iter = iter(cycle(data_io))
blob = next(data_iter)
for name,data in blob.items():
    print(name," : ",type(data)," -------------------")
    if type(data) is np.ndarray:
        print(data.shape," ",data.dtype)
