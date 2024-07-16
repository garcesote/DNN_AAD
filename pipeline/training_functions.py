import torch
import os
from utils.functional import get_subject, get_trials, normalize, correlation
from utils.datasets import FulsangDataset, HugoMapped
from utils.dnn import FCNN, CNN
from utils.ridge import Ridge
from torch.utils.data import Dataset, DataLoader
from torch.optim import NAdam
import torch.nn as nn
import numpy as np
import pickle
from collections.abc import Iterable
import operator
import functools
import scipy
import json

# SELECT NO RANDOM STATE
# torch.manual_seed(0)
# np.random.seed(0)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def train_dnn(model, dataset, data_path, metrics_save_path, date, mdl_save_path, max_epoch = 200, early_stopping_patience = 10):
    
    n_subjects = 18 if dataset=='Fulsang' else 13
    n_chan = 64 if dataset=='Fulsang' else 63
    batch_size = 125 if dataset=='Fulsang' else 256

    for n in range(n_subjects):

        subject = get_subject(n, n_subjects)

        print(f'Training {model} with {dataset} data on {subject}...')

        if model == 'FCNN':
            # FCNN (atención con el número de canales dependiendo del dataset)
            mdl = FCNN(n_hidden = 3, dropout_rate=0.45, n_chan=n_chan)
            optimizer = torch.optim.NAdam(mdl.parameters(), lr=1e-6, weight_decay = 1e-4)
        else:
            # CNN
            mdl = CNN(F1=8, D=8, F2=64, dropout=0.2, input_channels=n_chan)
            optimizer = torch.optim.NAdam(mdl.parameters(), lr=2e-5, weight_decay = 1e-8)

        mdl.to(device)

        if dataset == 'Fulsang':
            train_set = FulsangDataset(data_path, 'train', subject)
            train_loader = DataLoader(train_set, batch_size = batch_size, pin_memory=True)
            # train_loader = DataLoader(train_set, batch_size = batch_size, sampler = torch.randperm(len(train_set)), pin_memory=True)
            val_set = FulsangDataset(data_path, 'val', subject)
            val_loader = DataLoader(val_set, batch_size = batch_size, pin_memory=True)
            # val_loader = DataLoader(val_set, batch_size = batch_size, sampler = torch.randperm(len(val_set)), pin_memory=True)
        else:
            train_set = HugoMapped(range(9), data_path, participant=n)
            train_loader = DataLoader(train_set, batch_size = batch_size, pin_memory=True)
            # train_loader = DataLoader(train_set, batch_size = batch_size, sampler = torch.randperm(len(train_set)), pin_memory=True)
            val_set = HugoMapped(range(9, 12), data_path, participant=n)
            val_loader = DataLoader(val_set, batch_size = batch_size, pin_memory=True)
            # val_loader = DataLoader(val_set, batch_size = batch_size, sampler = torch.randperm(len(val_set)),pin_memory=True)

        # early stopping parameters
        best_accuracy=0
        best_epoch=0
        best_state_dict={}

        train_loss = []
        val_loss = []

        # training loop
        for epoch in range(max_epoch):
            
            # stop after n epoch without imporving the val loss
            if epoch > best_epoch + early_stopping_patience:
                break

            mdl.train()
            train_accuracies = []

            for batch, (x, y) in enumerate(train_loader):
                
                x = x.to(device, dtype=torch.float)
                y = y.to(device, dtype=torch.float)

                y_hat, loss = mdl(x, targets = y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_accuracies.append(- loss)

            mdl.eval()
            accuracies = []

            with torch.no_grad():

                for batch, (x,y) in enumerate(val_loader):
                    x = x.to(device, dtype=torch.float)
                    y = y.to(device, dtype=torch.float)

                    y_hat, loss = mdl(x)
                    accuracies.append(correlation(y, y_hat))

            mean_accuracy = torch.mean(torch.hstack(accuracies)).item()
            mean_train_accuracy = torch.mean(torch.hstack(train_accuracies)).item()
            print(f'Epoch: {epoch} | train accuracy: {mean_train_accuracy} | val accuracy: {mean_accuracy}')

            train_loss.append(mean_train_accuracy)
            val_loss.append(mean_accuracy)

            # Save best results
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_epoch = epoch
                best_state_dict = mdl.state_dict()

        if dataset == 'Fulsang':
            folder = 'fulsang_64Hz_data'
        else:
            folder = 'hugo_data'

        # save best final model
        torch.save(
            best_state_dict, 
            os.path.join(mdl_save_path, folder, model+'_'+date+'_'+subject+f'_epoch={epoch}_acc={mean_accuracy:.4f}.ckpt')
        )

        # save corresponding metrics
        json.dump(train_loss, open(os.path.join(metrics_save_path, folder, model+'_'+date+'_'+subject+'_train_loss'),'w'))
        json.dump(val_loss, open(os.path.join(metrics_save_path, folder, model+'_'+date+'_'+subject+'_val_loss'),'w'))

def train_ridge(dataset, data_path, mdl_save_path, date, start_lag=0, end_lag=50, original=False):

    # FOR ALL SUBJECTS
    n_subjects = 18 if dataset == 'Fulsang' else 13
    batch_size = 128 if dataset == 'Fulsang' else 256
    alphas = np.logspace(-7,7, 15)

    for n in range(n_subjects):

        subject = get_subject(n, n_subjects)
        
        mdl = Ridge(start_lag=start_lag, end_lag=end_lag, alpha=alphas, original=original)
        
        if dataset == 'Fulsang':
            train_set = FulsangDataset(data_path, 'train', subject)
            val_set = FulsangDataset(data_path, 'val', subject)
        else:
            train_set = HugoMapped(range(9), data_path, participant=n)
            val_set = HugoMapped(range(9, 12), data_path, participant=n)
        
        if dataset == 'Fulsang':
            train_eeg, train_stim = train_set.eeg, train_set.stima 
            val_eeg, val_stim = val_set.eeg, val_set.stima 
        else:
            train_eeg, train_stim = train_set.eeg, train_set.stim
            val_eeg, val_stim = val_set.eeg, val_set.stim
        
        # TRAIN MODEL
        mdl.fit(train_eeg.T, train_stim[:, np.newaxis])
        
        # VALIDATE AND SELECT BEST ALPHA
        scores = mdl.model_selection(val_eeg.T, val_stim[:, np.newaxis])
        best_alpha = mdl.best_alpha_idx
        print(f'Model for subject {n} trained with a score of {scores[best_alpha]} with alpha = {best_alpha}')

        # SAVE THE MODEL
        subj = get_subject(n, n_subjects)
        save_path = mdl_save_path + '/ridge/ridge_'+subj+'_'+date+'_'+dataset
        pickle.dump(mdl, open(save_path, "wb"))