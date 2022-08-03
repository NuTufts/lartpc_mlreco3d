from collections import defaultdict
from functools import wraps
import os
from tabnanny import verbose
import pandas as pd
from pprint import pprint

from mlreco.main_funcs import cycle
from mlreco.trainval import trainval
from mlreco.iotools.factories import loader_factory

from mlreco.utils.utils import ChunkCSVData

def recursive_merge(analysis_config, mlreco_config,
                    block=['schema', 'model', 'trainval'],
                    verbose=True):
    # Do not allow certain changes
    for fieldname in block:
        assert fieldname not in analysis_config
    for key in analysis_config:
        if key in mlreco_config:
            if isinstance(analysis_config[key], dict) and isinstance(mlreco_config[key], dict):
                recursive_merge(analysis_config[key], mlreco_config[key], block=block, verbose=verbose)
            else:
                assert type(analysis_config[key]) == type(mlreco_config[key])
                if verbose:
                    print("Override {} : {} -> {} : {}".format(
                        key, mlreco_config[key], key, analysis_config[key]))
                mlreco_config[key] = analysis_config[key]
        else:
            pass
    return mlreco_config


def evaluate(filenames, mode='per_image'):
    '''
    Inputs
    ------
        - analysis_function: algorithm that runs on a single image given by
        data_blob[data_idx], res
    '''
    def decorate(func):

        @wraps(func)
        def process_dataset(cfg, analysis_config):

            io_cfg = cfg['iotool']

            module_config = cfg['model']['modules']
            # # Override paths
            # cfg = recursive_merge(analysis_config, cfg)
            event_list = cfg['iotool']['dataset'].get('event_list', None)
            if event_list is not None:
                event_list = eval(event_list)
                if isinstance(event_list, tuple):
                    assert event_list[0] < event_list[1]
                    event_list = list(range(event_list[0], event_list[1]))

            loader = loader_factory(cfg, event_list=event_list)
            dataset = iter(cycle(loader))
            Trainer = trainval(cfg)
            loaded_iteration = Trainer.initialize()

            iteration = 0

            log_dir = analysis_config['analysis']['log_dir']
            append = analysis_config['analysis'].get('append', True)
            chunksize = analysis_config['analysis'].get('chunksize', 100)

            output_logs = []
            header_recorded = []

            for fname in filenames:
                fout = os.path.join(log_dir, fname + '.csv')
                output_logs.append(ChunkCSVData(fout, append=append, chunksize=chunksize))
                header_recorded.append(False)

            while iteration < analysis_config['analysis']['iteration']:
                data_blob, res = Trainer.forward(dataset)
                img_indices = data_blob['index']
                fname_to_update_list = defaultdict(list)
                if mode == 'per_batch':
                    # list of (list of dicts)
                    dict_list = func(data_blob, res, None, analysis_config, cfg)
                    for i, analysis_dict in enumerate(dict_list):
                        fname_to_update_list[filenames[i]].extend(analysis_dict)
                elif mode == 'per_image':
                    for batch_index, img_index in enumerate(img_indices):
                        dict_list = func(data_blob, res, batch_index, analysis_config, cfg)
                        for i, analysis_dict in enumerate(dict_list):
                            fname_to_update_list[filenames[i]].extend(analysis_dict)
                else:
                    raise Exception("Evaluation mode {} is invalid!".format(mode))
                for i, fname in enumerate(fname_to_update_list):
                    df = pd.DataFrame(fname_to_update_list[fname])
                    if len(df):
                        output_logs[i].record(df)
                        header_recorded[i] = True
                    # disable pandas from appending additional header lines
                    if header_recorded[i]: output_logs[i].header = False
                iteration += 1

        return process_dataset
    return decorate
