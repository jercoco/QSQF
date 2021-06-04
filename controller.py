#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 16:21:19 2021
control program execution flow
@author: wangke
"""

import os
import torch
import logging
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import TrainDataset,TestDataset,ValiDataset
from torch.utils.data.sampler import RandomSampler

import utils
from kernel import train_and_evaluate,evaluate
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', default='base_model', help='Dir of model')

def run(params,dirs,seed=None,restore_file=None):
    #set random seed to do reproducible experiments
    if seed is not None:
        utils.seed(seed)
    utils.set_logger(os.path.join(dirs.model_dir, 'train.log'))
    logger = logging.getLogger('DeepAR.Train')
    #check cuda is avaliable or not
    use_cuda=torch.cuda.is_available()
    # Set random seeds for reproducible experiments if necessary
    if use_cuda:
        dirs.device = torch.device('cuda:0')
        logger.info('Using Cuda...')
        model = net.Net(params,dirs.device).cuda(dirs.device)
    else:
        dirs.device = torch.device('cpu')
        logger.info('Not using cuda...')
        model = net.Net(params,dirs.device)

    logger = logging.getLogger('DeepAR.Data')
    logger.info('Loading the datasets...')
    train_set = TrainDataset(dirs.data_dir, dirs.dataset)
    vali_set = ValiDataset(dirs.data_dir, dirs.dataset)
    test_set = TestDataset(dirs.data_dir, dirs.dataset)
    train_loader = DataLoader(train_set,batch_size=params.batch_size,pin_memory=False,
                              num_workers=4)
    vali_loader = DataLoader(vali_set,batch_size=params.batch_size,pin_memory=False,
                             sampler=RandomSampler(vali_set),num_workers=4)
    test_loader = DataLoader(test_set,batch_size=params.batch_size,pin_memory=False,
                             sampler=RandomSampler(test_set),num_workers=4)
    logger.info('Data loading complete.')
    logger.info('###############################################\n')

    logger = logging.getLogger('DeepAR.Train')
    logger.info(f'Model: \n{str(model)}')
    logger.info('###############################################\n')

    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    # fetch loss function
    loss_fn = net.loss_fn
    # Train the model
    logger.info('Starting training for {} epoch(s)'.format(params.num_epochs))
    train_and_evaluate(model,train_loader,vali_loader,optimizer,loss_fn,scheduler,params,dirs,restore_file)
    logger.handlers.clear()
    logging.shutdown()

    load_dir = os.path.join(dirs.model_save_dir, 'best.pth.tar')
    if not os.path.exists(load_dir):
        return
    utils.load_checkpoint(load_dir, model)
    out=evaluate(model, loss_fn, test_loader, params, dirs, istest=True)
    test_json_path=os.path.join(dirs.model_dir,'test_results.json')
    utils.save_dict_to_json(out, test_json_path)

if __name__ == '__main__':
    args = parser.parse_args()
    params_path=os.path.join(args.model_dir,'params.json')
    dirs_path=os.path.join(args.model_dir,'dirs.json')
    params = utils.Params(params_path)
    dirs=utils.Params(dirs_path)
    if params.line=='QAspline':
        import model.net_qspline_A as net
    elif params.line=='QBspline':
        import model.net_qspline_B as net
    elif params.line=='QABspline':
        import model.net_qspline_AB as net
    elif params.line=='QCDspline':
        import model.net_qspline_C as net
    elif params.line=='Lspline':
        import model.net_lspline as net
    run(params,dirs)
