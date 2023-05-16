import os, sys, glob, inspect
import numpy as np
from torch.utils.data import Dataset
import mlreco.iotools.parser_factory as parser_factory
import mlreco.iotools.parsers

class LArCVDataset(Dataset):
    """
    A generic interface for LArCV data files.

    This Dataset is designed to produce a batch of arbitrary number
    of data chunks (e.g. input data matrix, segmentation label, point proposal target, clustering labels, etc.).
    Each data chunk is processed by parser functions defined in the iotools.parsers module. LArCVDataset object
    can be configured with arbitrary number of parser functions where each function can take arbitrary number of
    LArCV event data objects. The assumption is that each data chunk respects the LArCV event boundary.
    """
    def __init__(self, data_schema, data_keys, limit_num_files=0, limit_num_samples=0,
                 event_list=None, skip_event_list=None, nvoxel_limit=-1,
                 apply_crop=False,crop_cfg=None,use_alt_data_list=None):
        """
        Instantiates the LArCVDataset.

        Parameters
        ----------
        data_schema : dict
            A dictionary of (string, dictionary) pairs. The key is a unique name of
            a data chunk in a batch and the associated dictionary must include:
              - parser: name of the parser
              - args: (key, value) pairs that correspond to parser argument names and their values
            The nested dictionaries can replaced be lists, in which case
            they will be considered as parser argument values, in order.
        data_keys : list
            a list of strings that is required to be present in the file paths
        limit_num_files : int
            an integer limiting number of files to be taken per data directory
        limit_num_samples : int
            an integer limiting number of samples to be taken per data
        event_list : list
            a list of integers to specify which event (ttree index) to process
        skip_event_list : list
            a list of integers to specify which events (ttree index) to skip
        nvoxel_limit: int
            the limit to the number of voxels in the tensor, i.e. cap on length of coord and feat tensors
        apply_crop: bool
            if true, apply spatial cropping of the voxels. default: False
        crop_cfg: dict
            if apply_crop=True, will need the configuration for the cropper. 
            See function for parameters: crop_around_neutrino_voxels.
        """

        # Create file list
        self._files = []
        for key in data_keys:
            fs = glob.glob(key)
            for f in fs:
                self._files.append(f)
                if len(self._files) >= limit_num_files: break
            if len(self._files) >= limit_num_files: break

        if len(self._files)<1:
            print("ERROR: Number of files loaded is zero.")
            raise FileNotFoundError
        elif len(self._files)>10: print(len(self._files),'files loaded')
        else:
            for f in self._files: print('Loading file:',f)

        # Instantiate parsers
        self._data_keys = []
        self._data_parsers = []
        self._trees = {}
        for key, value in data_schema.items():
            # Check that the schema is a dictionary
            if not isinstance(value, dict):
                raise ValueError('A data schema must be expressed as a dictionary')

            # Identify the parser and its parameter names
            assert 'parser' in value, 'A parser needs to be specified for %s' % key
            if not hasattr(mlreco.iotools.parsers, value['parser']):
                print('The specified parser name %s does not exist!' % value['parser'])
            assert 'args' in value, 'Parser arguments must be provided for %s' % key
            fn = getattr(mlreco.iotools.parsers, value['parser'])
            keys = list(inspect.signature(fn).parameters.keys())
            assert isinstance(value['args'], dict), 'Parser arguments must be a list or dictionary for %s' % key
            for k in value['args'].keys():
                assert k in keys, 'Argument %s does not exist in parser %s' % (k, value['parser'])

            # Append data key and parsers
            self._data_keys.append(key)


            # BY TMW TO GET UBOONE PARSER IN
            # parser = None
            # try:
            #     parser = getattr(mlreco.iotools.parsers,value[0])
            # except:
            #     parser = None

            # if parser is None:
            #     print("Trying to get parser '%s' from parser_factory"%(value[0]))
            #     parser = parser_factory.get_parser( value[0] )
                                
            self._data_parsers.append((getattr(mlreco.iotools.parsers, value['parser']), value['args']))

            for arg_name, data_key in value['args'].items():
                if 'event' not in arg_name: continue
                if 'event_list' not in arg_name: data_key = [data_key]
                for k in data_key:
                    if k not in self._trees: self._trees[k] = None

        self._data_keys.append('index')

        # Prepare TTrees and load files
        from ROOT import TChain
        self._entries = None
        for data_key in self._trees.keys():
            # Check data TTree exists, and entries are identical across >1 trees.
            # However do NOT register these TTrees in self._trees yet in order to support >1 workers by DataLoader
            print('Loading tree',data_key)
            chain = TChain(data_key + "_tree")
            for f in self._files:
                chain.AddFile(f)
            if self._entries is not None: assert(self._entries == chain.GetEntries())
            else: self._entries = chain.GetEntries()

        # If event list is provided, register
        if event_list is None:
            self._event_list = np.arange(0, self._entries)
        elif isinstance(event_list, tuple):
            event_list = np.arange(event_list[0], event_list[1])
            self._event_list = event_list
            self._entries = len(self._event_list)
        else:
            if isinstance(event_list,list): event_list = np.array(event_list).astype(np.int32)
            assert(len(event_list.shape)==1)
            where = np.where(event_list >= self._entries)
            removed = event_list[where]
            if len(removed):
                print('WARNING: ignoring some of specified events in event_list as they do not exist in the sample.')
                print(removed)
            self._event_list = event_list[np.where(event_list < self._entries)]
            self._entries = len(self._event_list)

        if skip_event_list is not None:
            self._event_list = self._event_list[~np.isin(self._event_list, skip_event_list)]
            self._entries = len(self._event_list)

        # Set total sample size
        if limit_num_samples > 0 and self._entries > limit_num_samples:
            self._entries = limit_num_samples

        self._nvoxel_limit = nvoxel_limit

        # Cropper configuration
        self.apply_crop = apply_crop
        self.crop_cfg   = crop_cfg

        print('Found %d events in file(s)' % len(self._event_list))

        # Flag to identify if Trees are initialized or not
        self._trees_ready=False

    @staticmethod
    def list_data(f):
        from ROOT import TFile
        f=TFile.Open(f,"READ")
        data={'sparse3d':[],'cluster3d':[],'particle':[]}
        for k in f.GetListOfKeys():
            name = k.GetName()
            if not name.endswith('_tree'): continue
            if not len(name.split('_')) < 3: continue
            key = name.split('_')[0]
            if not key in data.keys(): continue
            data[key] = name[:name.rfind('_')]
        return data

    @staticmethod
    def get_event_list(cfg, key):
        event_list = None
        if key in cfg:
            if os.path.isfile(cfg[key]):
                event_list = [int(val) for val in open(cfg[key],'r').read().replace(',',' ').split() if val.isdigit()]
            else:
                try:
                    import ast
                    event_list = ast.literal_eval(cfg[key])
                except SyntaxError:
                    print('iotool.dataset.%s has invalid representation:' % key,event_list)
                    raise ValueError
        return event_list

    @staticmethod
    def create(cfg):
        data_schema = cfg['schema']
        if 'alt_data_list' in cfg:
            print("CREATE LArCVDataset using ALT data list: ",cfg['alt_data_list'])
            data_keys = cfg[cfg['alt_data_list']]
        else:
            print("CREATE LArCVDataset")
            data_keys   = cfg['data_keys']
        lnf         = 0 if not 'limit_num_files' in cfg else int(cfg['limit_num_files'])
        lns         = 0 if not 'limit_num_samples' in cfg else int(cfg['limit_num_samples'])
        event_list  = LArCVDataset.get_event_list(cfg, 'event_list')
        skip_event_list = LArCVDataset.get_event_list(cfg, 'skip_event_list')
        nvoxel_limit = -1 if not 'nvoxel_limit' in cfg else int(cfg['nvoxel_limit'])
        docrop       = bool(cfg['apply_crop'])
        if docrop:
            crop_cfg = cfg.get('crop_cfg',{})
        else:
            crop_cfg = None

        return LArCVDataset(data_schema=data_schema,
                            data_keys=data_keys,
                            limit_num_files=lnf,
                            event_list=event_list,
                            skip_event_list=skip_event_list,
                            nvoxel_limit=nvoxel_limit,
                            apply_crop=docrop,
                            crop_cfg=crop_cfg)

    def data_keys(self):
        return self._data_keys

    def __len__(self):
        return self._entries

    def __getitem__(self,idx):

        # convert to actual index: by default, it is idx, but not if event_list provided
        event_good = False
        ii = 0
        while not event_good:

            iidx = idx+ii
            if iidx>len(self._event_list):
                ii  = 0
                idx = 0
                iidx = 0
            event_idx = self._event_list[iidx]

            # If this is the first data loading, instantiate chains
            if not self._trees_ready:
                from ROOT import TChain
                for key in self._trees.keys():
                    chain = TChain(key + '_tree')
                    for f in self._files: chain.AddFile(f)
                    self._trees[key] = chain
                self._trees_ready=True

            # Move the event pointer
            for tree in self._trees.values():
                tree.GetEntry(event_idx)

            # Create data chunks
            result = {}
            for index, (parser, args) in enumerate(self._data_parsers):
                kwargs = {}
                for k, v in args.items():
                    if   'event_list' in k:
                        kwargs[k] = [getattr(self._trees[vi], vi+'_branch') for vi in v]
                    elif 'event' in k:
                        kwargs[k] = getattr(self._trees[v], v+'_branch')
                    else:
                        kwargs[k] = v
                name = self._data_keys[index]
                result[name] = parser(**kwargs)

            # Optional: Crop around the neutrino voxels
            if self.apply_crop:
                self.crop_around_neutrino_voxels( result )
            
            # Limit the number of voxels in the tensor.
            # if over, sample randomly
            #print(result)
            coord = result['input_data'][0]
            nvoxels = coord.shape[0]
            #print("NVOXELS: ",nvoxels)
            #sys.stdout.flush()
            
            if self._nvoxel_limit>0 and nvoxels>self._nvoxel_limit:
                self.resample_for_voxel_limit( result )

            if nvoxels>50:
                event_good = True
            else:
                event_good = False
                ii += 1
                print("RE-DRAW")
            result['index'] = event_idx
        return result

    def resample_for_voxel_limit( self, result ):
        """
        modifies
        --------
        result: dict
           contains the different torch tensor products        
        """
        print("result keys: ",result.keys())
        sys.stdout.flush()
        print("hit voxel limit [",nvoxels,">",self._nvoxel_limit," subsampling required")
        # we need to subsample
        subsample_fraction = float(self._nvoxel_limit)/float(nvoxels)
        r = np.random.rand( nvoxels )
        sel = r<subsample_fraction
        print("sample down to ",sel.sum()," voxels")            
        for x in result:
            coord = result[x][0][sel[:]]
            feat  = result[x][1][sel[:]]
            result[x] = (coord,feat)
        return True
        
    def crop_around_neutrino_voxels( self, result ):
        """
        We first throw a random number to determine if we crop randomly or specifically around neutrino voxels.
        
        If we crop around neutrino voxels, we find them using the keypoint labels, which mark the vertex.
        We center the crop roughly around the location.
        
        """

        #print("RUN CROPPER: crop_cfg=",self.crop_cfg)
        
        keypoint_treename = self.crop_cfg.get('nu_voxel_label','keypoint_label')
        keypoint_index    = self.crop_cfg.get('nu_voxel_index',0)
        nu_prob           = self.crop_cfg.get('nu_prob',0.5)
        min_crop_voxel    = self.crop_cfg.get('min_crop_voxel',50)
        data_to_crop      = self.crop_cfg.get('data_to_crop',[])
        verbose           = self.crop_cfg.get('verbose',False)
        if len(data_to_crop)==0:
            raise ValueError("ERROR: list of tensor products to crop is zero. Set 'data_to_crop' to configure.")
        
        voxel_max = {0:1009,
                     1:781,
                     2:3457}
        crop_size = 512
        crop_half_size = int(crop_size/2)
        

        if verbose:
            print("////////////////////////")
            print(" PRE-CROP TENSORS")
            print("////////////////////////")
            for n in result:
                print("[",n,"]",type(result[n]))
                for i,t in enumerate(result[n]):
                    print("  [",i,"] ",t.shape)        
        
        keypoint_label_coords = result[keypoint_treename][0]
        keypoint_label_feats  = result[keypoint_treename][1]
        if verbose:
            print("keypoint_label_coords=",keypoint_label_coords.shape)
            print("keypoint_label_feats=",keypoint_label_feats.shape)
            
        nvoxels = keypoint_label_coords.shape[0]
        nnu_voxels = (keypoint_label_feats[:,0]>0.1).sum()
        nu_crop = False
        if nnu_voxels>0:
            nu_crop = True # enough voxels to perform a nu crop
        if np.random.uniform()>nu_prob:
            # if we want to not crop around the nu interaction sometime
            nu_crop = False

        if verbose:
            print("we crop around nu voxels: ",nu_crop)

        crop_ok = False
        ntries = 0
        while not crop_ok and ntries<100:
            # set the crop center
            crop_center = [0,0,0]
            if nu_crop:
                # crop around nu voxels
                pos_info = {}
                for i in range(3):
                    nu_labels = keypoint_label_coords[ keypoint_label_feats[:,0]>0.1, i ]
                    pos_info[i] = nu_labels.mean()
                    if verbose:
                        print("  dim[",i,"] nu_labels.shape=",nu_labels.shape," mean=",pos_info[i])
                    crop_center[i] = int(pos_info[i]) + int(crop_size*(np.random.uniform()-0.5))
                    if crop_center[i]-crop_half_size<0:
                        crop_center[i] = crop_half_size
                    if crop_center[i]+crop_half_size>voxel_max[i]:
                        crop_center[i] = voxel_max[i]-(crop_half_size+1)
            else:
                # random crop inside, choose voxel with charge
                randidx = np.random.randint(0,nvoxels)
                for i in range(3):
                    crop_center[i] = keypoint_label_coords[randidx,i]
                    if crop_center[i]-crop_half_size<0:
                        crop_center[i] = crop_half_size
                    if crop_center[i]+crop_half_size>voxel_max[i]:
                        crop_center[i] = voxel_max[i]-(crop_half_size+1)
            if verbose:
                print("crop center: ",crop_center)

            # mark the voxels to include
            isinside = {}
            for i in range(3):
                bounded_low  = keypoint_label_coords[:,i]>=(crop_center[i]-crop_half_size)
                bounded_high = keypoint_label_coords[:,i]<(crop_center[i]+crop_half_size)
                isinside[i] = bounded_low*bounded_high
            bounded_index = isinside[0]*isinside[1]*isinside[2]
            ninside = bounded_index.sum()
            if verbose:
                print("number of bounded voxels: ",ninside)            
                print("bounded_index.shape=",bounded_index.shape)
            if ninside>=min_crop_voxel:
                crop_ok = True
            ntries += 1
        # end of while loop to find the bounds

        # apply the bounds
        for n in result:
            if n not in data_to_crop:
                continue
            if verbose:
                print("crop data name=",n)
            tensor_list = result[n]
            out_list = []
            for t in tensor_list:
                x = t[ bounded_index[:], : ]
                out_list.append(x)
            result[n] = (out_list[0],out_list[1])

        if verbose:
            print("////////////////////////")
            print(" POST-CROP TENSORS")
            print("////////////////////////")
            for n in result:
                print("[",n,"]")
                for i,t in enumerate(result[n]):
                    print("  [",i,"] ",t.shape)
            
