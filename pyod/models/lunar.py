# -*- coding: utf-8 -*-
"""Example of using LUNAR for outlier detection
detection
"""
# Author: Adam Goodge <a.goodge@u.nus.edu>
#

import os
from copy import deepcopy
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import faiss
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base import BaseDetector

# negative samples to train LUNAR
def generate_negative_samples(x, sample_type, proportion, epsilon):
    
    n_samples = int(proportion*(len(x)))
    n_dim = x.shape[-1]
    
    # uniform samples
    rand_unif = np.random.rand(n_samples,n_dim).astype('float32')
    #  subspace perturbation samples
    x_temp = x[np.random.choice(np.arange(len(x)),size = n_samples)]
    randmat = np.random.rand(n_samples,n_dim) < 0.3
    rand_sub = x + randmat*(epsilon*np.random.randn(n_samples,n_dim)).astype('float32')
    
    if sample_type == 'UNIFORM':
        neg_x = rand_unif
    if sample_type == 'SUBSPACE':
        neg_x = rand_sub
    if sample_type == 'MIXED':
        # randomly sample from uniform and gaussian negative samples
        neg_x = np.concatenate((rand_unif, rand_sub),0)
        neg_x = neg_x[np.random.choice(np.arange(len(neg_x)), size = n_samples)]

    neg_y = np.ones(len(neg_x))
    
    return neg_x.astype('float32'), neg_y.astype('float32')

class GNN(torch.nn.Module):
        def __init__(self,k):
            super(GNN, self).__init__()
            self.hidden_size = 256
            self.network = nn.Sequential(
                nn.Linear(k,self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size,self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size,self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size,1),
                nn.Sigmoid()
                )

        def forward(self,x):
            out = self.network(x)
            out = torch.squeeze(out,1)
            return out

class LUNAR(BaseDetector):
    def __init__(self):
        super(LUNAR, self).__init__()
        self.args = {
        # n_neighbors
        "k": 5,
        # negative sampling type
        "negative_sampling": 'MIXED',
        # negative sampling parameter
        "epsilon": 0.1,
        # ratio of negative samples to normal samples
        "proportion": 1,
        "n_epochs": 200,
        # learning rate
        "lr": 0.001,
        # weight decay
        "wd": 0.1,  
        "verbose": 0,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
        }

        self.network = GNN(self.args['k']).to(self.args['device'])

    def fit(self, X, y=None):
        
        if y is None:
            y = np.zeros(len(X))
        #split  train/val
        train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.2)
        neg_train_x, neg_train_y = generate_negative_samples(train_x,self.args['negative_sampling'],self.args['proportion'],self.args['epsilon'])
        neg_val_x, neg_val_y = generate_negative_samples(val_x,self.args['negative_sampling'],self.args['proportion'],self.args['epsilon'])

        # concat data
        x = np.vstack((train_x,neg_train_x,val_x,neg_val_x))
        y = np.hstack((train_y,neg_train_y,val_y,neg_val_y))

        # all training set
        train_mask = np.hstack((np.ones(len(train_x)),np.ones(len(neg_train_x)),
                                np.zeros(len(val_x)),np.zeros(len(neg_val_x))))

        # normal training points
        neighbor_mask = np.hstack((np.ones(len(train_x)), np.zeros(len(neg_train_x)), 
                                np.zeros(len(val_y)), np.zeros(len(neg_val_x))))
                                
        # nearest neighbour object
        self.neigh = faiss.IndexFlatL2(x.shape[-1])
        # add nearest neighbour candidates using neighbour mask
        self.neigh.add(x[neighbor_mask==1])

        # distances and idx of neighbour points for the neighbour candidates (k+1 as the first one will be the point itself)
        dist_train, idx_train = self.neigh.search(x[neighbor_mask==1], k = self.args['k']+1)
        # remove 1st nearest neighbours to remove self loops
        dist_train, idx_train = dist_train[:,1:], idx_train[:,1:]
        # distances and idx of neighbour points for the non-neighbour candidates
        dist, idx = self.neigh.search(x[neighbor_mask==0], k = self.args['k'])
        #concat
        dist = np.sqrt(np.vstack((dist_train, dist)))
        idx = np.vstack((idx_train, idx))

        dist = torch.tensor(dist,dtype=torch.float32).to(self.args['device'])
        y = torch.tensor(y,dtype=torch.float32).to(self.args['device'])

        criterion = nn.MSELoss(reduction = 'none')    
        
        optimizer = optim.Adam(self.network.parameters(), lr = self.args['lr'], weight_decay = self.args['wd'])
        
        best_val_score = 0
        
        for epoch in range(self.args['n_epochs']):

            # see performance of model on each set before each epoch
            with torch.no_grad():
                
                self.network.eval()
                out = self.network(dist)
                out = out.cpu()
                loss = criterion(out,y.cpu())
                train_score = roc_auc_score(y[train_mask==1].cpu(), out[train_mask==1])
                val_score = roc_auc_score(y[train_mask == 0].cpu(), out[train_mask == 0])

                # save best model 
                if val_score >= best_val_score:  
                    best_dict = {'epoch': epoch,
                                'model_state_dict': deepcopy(self.network.state_dict()),
                                'optimizer_state_dict': deepcopy(optimizer.state_dict()),
                                'train_score': train_score,
                                'val_score': val_score,
                                }

                    # reset best score so far
                    best_val_score = val_score
                
                if self.args['verbose'] == 1:
                    print(f"Epoch {epoch} \t Train Score {np.round(train_score,6)} \t Val Score {np.round(val_score,6)}")

            #training
            self.network.train()
            optimizer.zero_grad()
            out = self.network(dist)
            loss = criterion(out[train_mask == 1],y[train_mask == 1]).sum()
            loss.backward()
            optimizer.step()
        
        if self.args['verbose'] == 1:
            print(f"Epoch {best_dict['epoch']} Train Score {best_dict['train_score']} Val Score {best_dict['val_score']}")

        # load best model
        self.network.load_state_dict(best_dict['model_state_dict'])

        return self

    def decision_function(self, X):
        dist, _ = self.neigh.search(X,self.args['k'])
        dist = torch.tensor(dist,dtype=torch.float32).to(self.args['device'])
        with torch.no_grad():
            self.network.eval()
            out = self.network(dist)
        
        # return output for test points
        return out.cpu().detach().numpy() 