#!/usr/bin/env python
import os
import sys
import yaml
from os import environ
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)
from mlreco.main_funcs import process_config, train, inference, set_cuda_visible_devices
import mlreco.iotools.ubparsers

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size, backend, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'
    os.environ['MASTER_PORT'] = port
    # initialize the process group
    #dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    # this function is responsible for synchronizing and successfully communicate across multiple process
    # involving multiple GPUs.

def main(rank, run_parallel, config):

    # Load configuration file
    cfg_file = config
    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(current_directory, 'config', config)
    if not os.path.isfile(cfg_file):
        print(config, 'not found...')
        sys.exit(1)

    cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)

    # process the configuration
    process_config(cfg)

    # add rank
    cfg['rank'] = rank

    # get number of gpus to use
    ngpus = len(cfg['trainval']['gpus'])

    # set gpu(s)
    if not cfg['ddp']:
        # will add available GPUs to be visible
        set_cuda_visible_devices(cfg)
    else:
        # in ddp mode, we have one process per gpu
        gpuid = int( cfg['trainval']['gpus'][rank] )        
        cfg['trainval']['gpus'] = [gpuid]
        # set the gpu for this process
        torch.cuda.set_device( gpuid )
        # and modify the seed
        cfg['trainval']['seed'] += cfg['rank']*10

        #========================================================
        # CREATE PROCESS, if using distributed data parallel
        print("[DDP MODE] START main() PROCESS: rank=%d ngpus=%d gpu=%d"%(rank,ngpus,gpuid))
        setup( rank, ngpus, "nccl", "12355" )
        #========================================================    

    if cfg['trainval']['train']:
        train(cfg)
    else:
        inference(cfg)

    if cfg['ddp']:
        cleanup()

if __name__ == '__main__':
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument("--detect_anomaly",
                        help="Turns on autograd.detect_anomaly for debugging",
                        action='store_true')
    args = parser.parse_args()

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # LOAD THE CONFIG: WE NEED THE NUMBER OF GPUs
    cfg_file = args.config
    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(current_directory, 'config', config)
    if not os.path.isfile(cfg_file):
        print(config, 'not found...')
        sys.exit(1)

    cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)

    if environ.get('CUDA_VISIBLE_DEVICES') is not None and cfg['trainval']['gpus'] == '-1':
        cfg['trainval']['gpus'] = os.getenv('CUDA_VISIBLE_DEVICES')

    process_config(cfg)
    
    if not cfg['ddp']:
        main(0,False,args.config,)
    else:
        ngpus = len(cfg['trainval']['gpus'])
        print("RUN WITH TORCH DDP: NGPUS=",ngpus)        
        mp.spawn(main, nprocs=ngpus, args=(True,args.config,), join=True)
            
