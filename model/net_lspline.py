# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 19:52:22 2020

@author: 18096
"""

'''Defines the neural network, loss function and metrics'''

import logging
import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch.autograd import Variable


logger = logging.getLogger('DeepAR.Net')


class Net(nn.Module):
    def __init__(self, params,device):
        '''
        We define a recurrent network that predicts the future values
        of a time-dependent variable based on past inputs and covariates.
        '''
        super(Net, self).__init__()
        self.params = params
        self.device=device
        self.lstm = nn.LSTM(input_size=params.lstm_input_size,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)
        # initialize LSTM forget gate bias to be 1 as recommanded by
        # http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        #self.relu = nn.ReLU()#TODO ReLU dose not used in forward
        self.spline_pre_u = nn.Linear(
            params.lstm_hidden_dim * params.lstm_layers, params.num_spline)
        self.spline_pre_beta = nn.Linear(
            params.lstm_hidden_dim * params.lstm_layers, params.num_spline)
        self.spline_gama = nn.Linear(
            params.lstm_hidden_dim * params.lstm_layers, 1)
        self.spline_u = nn.Softmax(dim=1)
        self.spline_beta = nn.Softplus()

    def forward(self, x, hidden, cell):
    #def forward(self,x,hidden,cell):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([1, batch_size, cov_dim]): z_{t-1} + x_t, note that z_0 = 0
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]):
            LSTM h from time step t-1
            cell ([lstm_layers, batch_size, lstm_hidden_dim]):
            LSTM c from time step t-1
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden ([lstm_layers, batch_size, lstm_hidden_dim]):
            LSTM h from time step t
            cell ([lstm_layers, batch_size, lstm_hidden_dim]):
            LSTM c from time step t
        '''
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        # use h from all three layers to calculate mu and sigma
        hidden_permute = \
            hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)

        #### settings of beta distribution
        pre_u = self.spline_pre_u(hidden_permute)
        # softmax to make sure Σu equals to 1
        spline_u = self.spline_u(pre_u)
        pre_beta = self.spline_pre_beta(hidden_permute)
        # softplus to make sure the intercept is positive
        spline_beta = self.spline_beta(pre_beta)
        spline_gama = self.spline_gama(hidden_permute)

        return ((spline_u,spline_beta,torch.squeeze(spline_gama)),\
            hidden, cell)

    def init_hidden(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size,
                           self.params.lstm_hidden_dim,
                           device=self.device)

    def init_cell(self, input_size):
        return torch.zeros(self.params.lstm_layers, input_size,
                           self.params.lstm_hidden_dim,
                           device=self.device)

    def predict(self, x, hidden, cell, sampling=False):
        """
        generate samples by sampling from
        """
        batch_size = x.shape[1]
        samples = torch.zeros(self.params.sample_times,batch_size,
                              self.params.pred_steps,
                              device=self.device)
        for j in range(self.params.sample_times):
            decoder_hidden = hidden
            decoder_cell = cell
            for t in range(self.params.pred_steps):
                func_param,decoder_hidden,decoder_cell=\
                    self(x[self.params.pred_start+t].unsqueeze(0),
                         decoder_hidden,decoder_cell)
                sigma,beta,gamma=func_param
                #pred_cdf is a uniform ditribution
                uniform = torch.distributions.uniform.Uniform(
                    torch.tensor([0.0], device=sigma.device),
                    torch.tensor([1.0], device=sigma.device))
                pred_cdf = torch.squeeze(uniform.sample([batch_size]))

                b = beta-pad(beta,(1,0),'constant',0)[:,:-1]
                d = torch.cumsum(pad(sigma,(1,0),'constant',0),dim=1)
                d = d[:, :-1]
                ind = d[:, 1:].T<pred_cdf
                ind = ind.T
                b[:,1:] = ind*b[:,1:]
                delta_x = pred_cdf - d.T
                delta_x = delta_x.T

                pred = gamma + torch.sum(b*delta_x, dim=1)
                samples[j, :, t] = pred
                #predict value at t-1 is as a covars for t,t+1,...,t+lag
                for lag in range(self.params.lag):
                    if t<self.params.pred_steps-lag-1:
                        x[self.params.pred_start+t+1,:,0]=pred

        sample_mu = torch.mean(samples, dim=0)  # mean or median ?
        sample_sigma = samples.std(dim=0)
        return samples, sample_mu, sample_sigma


def loss_fn(func_param, labels: Variable):
    sigma,beta,gamma=func_param
    #for each segment of the spline, the cumsum+gamma is the maximum
    knots = torch.cumsum(sigma*beta,dim=1).T + gamma
    knots = knots[:-1,:]
    l_0 = knots < labels
    #b_l=beta_l-beta_(l-1).
    b_l = beta - pad(beta,(1,0),'constant',0)[:,:-1]
    d_l = torch.cumsum(pad(sigma,(1,0),'constant',0),dim=1)
    d_l = d_l[:,:-1]
    bd = b_l*d_l
    denom = torch.sum(l_0.T*b_l[:,1:],dim=1) + b_l[:,0]
    pnom = torch.sum(l_0.T*bd[:,1:],dim=1) + bd[:,0]
    nom = labels - gamma + pnom
    alpha = nom/denom

    max_ad = d_l+(alpha>d_l.T).T*(alpha-d_l.T).T
    crps = (2*alpha - 1) * labels + (1 - 2*alpha) * gamma + \
           torch.sum(b_l*((1-d_l.pow(3))/3-d_l+max_ad*(2*d_l-max_ad)),dim=1)
    crps = torch.mean(crps)
    return crps
