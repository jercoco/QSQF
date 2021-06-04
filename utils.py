import os
import json
import math
import torch
import shutil
import logging
import matplotlib
import numpy as np
from tqdm import tqdm
import properscoring as ps
import matplotlib.pyplot as plt

from torch.backends import cudnn

matplotlib.use('Agg')
logger = logging.getLogger('DeepAR.Utils')
#matplotlib.rcParams['savefig.dpi'] = 300 #Uncomment for higher plot resolutions

def seed(seed):
    cudnn.enabled = False
    cudnn.benchmark = False# if benchmark=True, deterministic will be False
    cudnn.deterministic = True
    #set random seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)


class Params:
    '''Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    '''

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        '''Loads parameters from json file'''
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        '''Gives dict-like access to Params instance by params.dict['learning_rate']'''
        return self.__dict__


class RunningAverage:
    '''A simple class that maintains the running average of a quantity
    Example:
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    '''

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

def dirs_update(dirs):
    dirs.params_save_path=os.path.join(dirs.model_dir,'params.json')
    dirs.dirs_save_path=os.path.join(dirs.model_dir,'dirs.json')
    dirs.plot_dir=os.path.join(dirs.model_dir,'figures')
    dirs.model_save_dir=os.path.join(dirs.model_dir,'epochs')
    dirs.data_dir=os.path.join(dirs.data_dir,dirs.dataset)
    # create missing directories
    if not os.path.exists(dirs.plot_dir):
        os.makedirs(dirs.plot_dir)
    if not os.path.exists(dirs.model_save_dir):
        os.makedirs(dirs.model_save_dir)
    return dirs

def set_logger(log_path):
    '''Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    '''
    _logger = logging.getLogger('DeepAR')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    # _logger.addHandler(TqdmHandler(fmt))


def save_dict_to_json(d, json_path):
    '''Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    '''
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best,checkpoint,  ins_name=-1):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        ins_name: (int) instance index
    '''
    epoch_num=state['epoch']
    if ins_name == -1:
        #filepath = os.path.join(checkpoint, f'epoch_{epoch_num}.pth.tar')
        filepath = os.path.join(checkpoint, 'last.pth.tar')
    else:
        filepath = os.path.join(checkpoint, f'epoch_{epoch_num}_ins_{ins_name}.pth.tar')
    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    logger.info(f'Checkpoint saved to {filepath}')
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))



def load_checkpoint(checkpoint, model, optimizer=None):
    '''Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        gpu: which gpu to use
    '''
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def plot_train(variable, save_name, location='./figures/'):
    num_samples = variable.shape[0]
    x = np.arange(start=1, stop=num_samples + 1)
    f = plt.figure()
    plt.plot(x, variable[:num_samples])
    f.savefig(os.path.join(location, save_name + '_summary.png'))
    plt.close()

def init_metrics(params,dirs):
    metrics = {}
    shape_param=params.pred_steps
    device=dirs.device
    metrics['num']=torch.zeros(1,device=device)
    metrics['CRPS'] = torch.zeros(shape_param,device=device)
    metrics['mre']=torch.zeros((19,shape_param),device=device)
    metrics['pinaw'] = torch.zeros(shape_param,device=device)
    return metrics

def update_metrics(metrics, samples, labels, pred_start):
    df=labels[:, pred_start:]
    metrics['num']=metrics['num']+samples.shape[1]
    metrics['CRPS']=metrics['CRPS']+accuracy_CRPS(samples, df)
    metrics['mre']=metrics['mre']+accuracy_MRE(samples, df)
    metrics['pinaw'] = metrics['pinaw']+accuracy_PINAW(samples)
    return metrics

def final_metrics(metrics):
    summary = {}
    summary['CRPS'] = metrics['CRPS']/metrics['num']
    summary['mre']=metrics['mre']/metrics['num']
    summary['mre']=summary['mre'].T-torch.arange(0.05,1,0.05,device=metrics['mre'].device)
    summary['pinaw']=(metrics['pinaw']/metrics['num']).mean()
    return summary

def accuracy_CRPS(samples: torch.Tensor, labels: torch.Tensor):
    samples_permute=samples.permute(1,2,0)
    crps = ps.crps_ensemble(labels.cpu().detach().numpy(),
                                  samples_permute.cpu().detach().numpy()).sum(axis=0)
    return torch.Tensor(crps,device=samples.device)


def accuracy_MRE(samples: torch.Tensor, labels: torch.Tensor):
    samples_sorted = samples.sort(dim=0).values
    df1=torch.sum(samples_sorted>labels,dim=1)
    mre=df1[[i-1 for i in range(5,100,5)],:]
    return mre


def accuracy_PINAW(samples: torch.Tensor):
    out=torch.zeros(samples.shape[2],device=samples.device)
    for i in range(10,100,10):
        q_n1=samples.quantile(1-i/200,dim=0)
        q_n2=samples.quantile(i/200,dim=0)
        out=out+torch.sum((q_n1-q_n2)/(1-i/100),dim=0)
    out=out/9
    return out
