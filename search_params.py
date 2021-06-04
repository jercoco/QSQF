import os
import sys
#import math
import json
import logging
import multiprocessing
from copy import copy
from itertools import product
from subprocess import check_call

import numpy as np
import utils

import pickle
import time

logger = logging.getLogger('DeepAR.Searcher')
utils.set_logger('param_search.log')

PYTHON = sys.executable
gpu_ids: list
param_template: utils.Params
dirs:utils.Params
search_params: dict

def launch_training_job(search_range):
    """Launch training of the model with a set of hyper-parameters in parent_dir/job_name
    Args:
        search_range: one combination of the params to search
    """

    search_range = search_range[0]
    params = {k: search_params[k][search_range[idx]] for idx, k in enumerate(sorted(search_params.keys()))}
    model_param_list = '-'.join('_'.join((k, str(v))) for k, v in params.items())
    model_param = copy(param_template)
    for k, v in params.items():
        setattr(model_param, k, v)

    pool_id, job_idx = multiprocessing.Process()._identity
    gpu_id = gpu_ids[pool_id % 2]

    logger.info(f'Worker {pool_id} running {job_idx} using GPU {gpu_id}')

    dirs=copy(dirs_template)
    dirs.model_dir = os.path.join(dirs.model_dir,dirs.job_dir,model_param_list)
    dirs=utils.dirs_update(dirs)
    model_param.save(dirs.params_save_path)
    dirs.save(dirs.dirs_save_path)

    # Launch training with this config
    cmd = f'{PYTHON} controller.py '\
        f'--model-dir={dirs.model_dir}'

    logger.info(cmd)
    check_call(cmd, shell=True, env={'CUDA_VISIBLE_DEVICES': str(gpu_id),
                                     'OMP_NUM_THREADS': '4'})


def start_pool(project_list, processes):

    pool = multiprocessing.Pool(processes)
    pool.map(launch_training_job, [(i, ) for i in project_list])

def main():
    # Load the 'reference' parameters from parent_dir json file
    global param_template, gpu_ids, search_params, dirs_template

    time_start = time.time()
    stage=2#stage:1,2
    proc_num=10
    zones=['Zone'+str(i) for i in [1,2,3,4,5,6,7,8,9,10]]
    for zone in zones:
        json_path='./experiments/param_search/configuration/params.json'
        dirs_path='./experiments/param_search/configuration/dirs.json'
        param_template = utils.Params(json_path)
        dirs_template=utils.Params(dirs_path)
        dirs_template.dataset=zone
        dirs_template.model_dir=os.path.join(dirs_template.model_dir,dirs_template.model_name)
        dirs_template.job_dir=f'{stage}_{zone}'#TODO
        gpu_ids = dirs_template.gpu_ids
        logger.info(f'Running on GPU: {gpu_ids}')
        # Perform hypersearch over parameters listed below
        if stage==1:
            search_params = {
                #'lr':[0.001,0.0008,0.0006,0.0004],
                'line':['Lspline','QAspline','QABspline','QBspline','QCDspline']
            }
        elif stage==2:
            search_params = {'num_spline':[10,20,30,40,50,60,70,80,90,100],
                             'line':['QAspline','QABspline','QBspline','QCDspline']}

        keys = sorted(search_params.keys())
        search_range = list(product(*[[*range(len(search_params[i]))] for i in keys]))
        start_pool(search_range, len(gpu_ids)*proc_num)
        summary_path=os.path.join(dirs_template.model_dir,dirs_template.job_dir)

        vali_results = np.empty((34,len(search_range)))
        test_results = np.empty((34,len(search_range)))
        count = 0
        for i in search_range:
            params = {k: search_params[k][i[idx]] for idx, k in enumerate(sorted(search_params.keys()))}
            model_param_list = '-'.join('_'.join((k, str(v))) for k, v in params.items())
            model_name = os.path.join(summary_path, model_param_list)

            vali_json_path = os.path.join(model_name, 'vali_best.json')
            test_json_path = os.path.join(model_name, 'test_results.json')
            if os.path.exists(vali_json_path):
                with open(vali_json_path) as f:
                    temp = json.load(f)
                    vali_results[:, count] = np.array(list(temp.values()))
            else:
                vali_results[:,count] = np.nan

            if os.path.exists(test_json_path):
                with open(test_json_path) as f:
                    temp = json.load(f)
                    test_results[:, count] = np.array(list(temp.values()))
            else:
                test_results[:,count] = np.nan
            count+=1

        save_name = os.path.join(summary_path,'vali_summary')
        with open(save_name, 'wb') as f:
            pickle.dump(search_params, f)
            pickle.dump(vali_results, f)

        save_name = os.path.join(summary_path,'test_summary')
        with open(save_name, 'wb') as f:
            pickle.dump(search_params, f)
            pickle.dump(test_results, f)

    time_end = time.time()
    print('time cost:', (time_end - time_start)/60, 'min')


if __name__ == '__main__':
    main()
