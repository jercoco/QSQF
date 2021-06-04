
import os
import torch
import logging
import numpy as np
from tqdm import tqdm

import utils

logger = logging.getLogger('DeepAR.Train')


def train(model,
          optimizer,
          loss_fn,
          train_loader,
          params: utils.Params,
          dirs,
          epoch: int) -> float:
    '''Train the model on one epoch by batches.
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        train_loader: load train data and labels
        params: (Params) hyperparameters
        epoch: (int) the current training epoch
    '''
    model.train()#set model train mode
    loss_epoch = np.zeros(len(train_loader))
    flag=False
    # Train_loader:
    for i, (train_batch, labels_batch) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        batch_size = train_batch.shape[0]
        train_batch = train_batch.permute(1,0,2).\
            to(torch.float32).to(dirs.device)  # not scaled
        labels_batch = labels_batch.permute(1,0).\
            to(torch.float32).to(dirs.device)  # not scaled
        loss = torch.zeros(1, device=dirs.device,requires_grad=True)
        hidden = model.init_hidden(batch_size)
        cell = model.init_cell(batch_size)

        for t in range(params.train_window):
            func_param,hidden,cell=model(train_batch[t].unsqueeze_(0).clone(),hidden,cell)
            if torch.isnan(hidden).sum()>0:
                logger.info('Backward Error! Process Stop!')
                flag=True
                return (loss_epoch/params.train_window,flag)
            loss=loss+loss_fn(func_param,labels_batch[t])
            if torch.isnan(loss).sum()>0:
                logger.info(f'Loss Error at Data={i} Time={t}! Process Stop!')
                flag=True
                return (loss_epoch/params.train_window,flag)
        loss.backward()
        optimizer.step()
        loss_epoch[i] = loss.item()
    #output loss for per time
    return (loss_epoch/params.train_window,flag)


def evaluate(model, loss_fn, test_loader, params, dirs, istest=False):
    '''Evaluate the model on the test set.
    Args:
        model: (torch.nn.Module) the Deep AR model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        test_loader: load test data and labels
        params: (Params) hyperparameters
    '''
    logger = logging.getLogger('DeepAR.Eval')
    model.eval()
    with torch.no_grad():
        summary = {}
        metrics = utils.init_metrics(params,dirs)

        for i, (test_batch, labels) in enumerate(tqdm(test_loader)):
            test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(dirs.device)
            labels = labels.to(torch.float32).to(dirs.device)
            batch_size = test_batch.shape[1]
            hidden = model.init_hidden(batch_size)
            cell = model.init_cell(batch_size)

            for t in range(params.pred_start):
                _,hidden,cell=model(test_batch[t].unsqueeze(0),hidden,cell)

            #save some params of SQF for plot
            if istest and (i==0):
                plot_param,_,_=model(test_batch[params.pred_start].unsqueeze(0),hidden,cell)
                save_name=os.path.join(dirs.model_dir,'sqf_param')
                with open(save_name, 'wb') as f:
                    import pickle
                    pickle.dump(plot_param, f)
                    pickle.dump(test_batch[params.pred_start],f)

            samples,_,_=model.predict(test_batch,hidden,cell,sampling=True)
            metrics=utils.update_metrics(metrics,samples,labels,
                                         params.pred_start)
        summary = utils.final_metrics(metrics)
        if istest==False:
            strings ='\nCRPS: '+str(summary['CRPS'])+\
                        '\nmre:'+str(summary['mre'].abs().max(dim=1)[0].mean().item())+\
                            '\nPINAW:'+str(summary['pinaw'].item())
            logger.info('- Full test metrics: ' + strings)
        else:
            logger.info(' - Test Set CRPS: ' + str(summary['CRPS'].mean().item()))
    ss_metric = {}
    ss_metric['CRPS_Mean'] = summary['CRPS'].mean()
    ss_metric['mre'] = summary['mre'].abs().mean()
    ss_metric['pinaw'] = summary['pinaw']
    for i,crps in enumerate(summary['CRPS']):
        ss_metric[f'CRPS_{i}']=crps
    for i,mre in enumerate(summary['mre'].mean(dim=0)):
        ss_metric[f'mre_{i}']=mre
    return ss_metric


def train_and_evaluate(model,
                       train_loader,
                       test_loader,
                       optimizer,
                       loss_fn,
                       params: utils.Params,
                       dirs: utils.Params,
                       restore_file):
    '''Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the Deep AR model
        train_loader: load train data and labels
        test_loader: load test data and labels
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes outputs and labels per timestep, and then computes the loss for the batch
        params: (Params) hyperparameters
        args:
    '''
    best_test_CRPS = (int(0), float('inf'))
    train_len = len(train_loader)
    CRPS_summary = np.zeros(params.num_epochs)
    PINAW_summary = np.zeros(params.num_epochs)
    rc_summary = np.zeros(params.num_epochs)
    loss_summary = np.zeros((train_len * params.num_epochs))
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(dirs.model_dir,'epochs','epoch_'+str(restore_file)+'.pth.tar')
        logger.info('Restoring parameters from {}'.format(restore_path))
        checkpoint=utils.load_checkpoint(restore_path,model,optimizer)
        current_epoch=checkpoint['epoch']
        best_test_CRPS=checkpoint['best_test_CRPS']
        CRPS_summary=checkpoint['CRPS_summary']
        rc_summary=checkpoint['rc_summary']
        loss_summary=checkpoint['loss_summary']
    logger.info('Begin training and evaluation')
    for epoch in range(params.num_epochs):
        if restore_file is not None:
            if epoch<current_epoch:
                continue
        logger.info('----------------------------------------------\n')
        logger.info('Epoch {}/{}'.format(epoch + 1, params.num_epochs))
        lr=optimizer.param_groups[0]['lr']
        logger.info('learning rate is: '+str(lr))
        loss,flag=train(model,optimizer,loss_fn,train_loader,params,dirs,epoch)
        loss_summary[epoch*train_len:(epoch+1)*train_len]=loss
        if flag:
            break
        test_metrics = evaluate(model,loss_fn,test_loader,params,dirs)
        CRPS_summary[epoch] = test_metrics['CRPS_Mean']
        PINAW_summary[epoch] = test_metrics['pinaw']
        rc_summary[epoch]=test_metrics['mre']
        is_best = CRPS_summary[epoch] <= best_test_CRPS[1]

        # Save weights
        dict_to_save={'epoch': epoch+1,
                      'state_dict': model.state_dict(),
                      'optim_dict': optimizer.state_dict(),
                      'best_test_CRPS': best_test_CRPS,
                      'CRPS_summary':CRPS_summary,
                      'rc_summary':rc_summary,
                      'loss_summary':loss_summary}
        utils.save_checkpoint(dict_to_save,is_best,checkpoint=dirs.model_save_dir)

        if is_best:
            logger.info('####Found new best CRPS')
            best_test_CRPS = (epoch+1, CRPS_summary[epoch])
            best_json_path = os.path.join(dirs.model_dir, 'vali_best.json')
            utils.save_dict_to_json(test_metrics, best_json_path)

        logger.info('Current Best CRPS is: %.5f of epoch %d'%(best_test_CRPS[1], best_test_CRPS[0]))

        utils.plot_train(CRPS_summary[:(epoch + 1)], dirs.dataset + '_CRPS', dirs.plot_dir)
        utils.plot_train(PINAW_summary[:(epoch + 1)], dirs.dataset + '_PINAW', dirs.plot_dir)
        utils.plot_train(rc_summary[:(epoch + 1)], dirs.dataset + '_MRE', dirs.plot_dir)
        utils.plot_train(loss_summary[:(epoch + 1) * train_len], dirs.dataset + '_loss', dirs.plot_dir)

        last_json_path = os.path.join(dirs.model_dir, 'vali_last.json')
        utils.save_dict_to_json(test_metrics, last_json_path)

        #stop the iteration
        if (epoch-best_test_CRPS[0])>=params.max_delay_epochs:
            logger.info('CRPS do not decrease anymore!')
            break
    return best_test_CRPS
