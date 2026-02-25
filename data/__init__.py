from functools import partial
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch import Generator
from torch.utils.data import DataLoader

import core.util as Util
from core.praser import init_obj


def define_dataloader(logger, opt):
    """ create train/test dataloader and validation dataloader """
    dataloader_args = opt['datasets'][opt['phase']]['dataloader']['args']
    worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])

    phase_dataset, val_dataset = define_dataset(logger, opt)

    '''create datasampler'''
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(phase_dataset, shuffle=dataloader_args.get('shuffle', False), num_replicas=opt['world_size'], rank=opt['global_rank'])
        dataloader_args.update({'shuffle':False})
    
    ''' create dataloader and validation dataloader '''
    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    
    ''' val_dataloader don't use DistributedSampler to run only GPU 0! '''
    if opt['global_rank']==0 and val_dataset is not None:
        val_dataloader_args = opt['datasets']['val']['dataloader']['args']
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **val_dataloader_args) 
    else:
        val_dataloader = None
        
    return dataloader, val_dataloader


def define_dataset(logger, opt):
    ''' loading Dataset() class from given file's name '''
    dataset_opt = opt['datasets'][opt['phase']]['which_dataset']
    phase_dataset = init_obj(dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')
    val_dataset = None

    # Chargement explicite du dataset de validation si on est en phase d'entraînement
    if opt['phase'] == 'train' and 'val' in opt['datasets']:
        val_dataset_opt = opt['datasets']['val']['which_dataset']
        val_dataset = init_obj(val_dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')
    
    logger.info('Dataset for {} has {} samples.'.format(opt['phase'], len(phase_dataset)))
    if val_dataset is not None:
        logger.info('Dataset for val has {} samples.'.format(len(val_dataset)))   
        
    return phase_dataset, val_dataset