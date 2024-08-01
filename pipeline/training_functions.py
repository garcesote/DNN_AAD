import torch
import os
from utils.functional import get_subject, correlation, get_params, check_jaulab_chan, get_Dataset
from utils.dnn import FCNN, CNN
from utils.ridge import Ridge
from utils.datasets import JaulabDatasetWindows, FulsangDatasetWindows
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import pickle
from statistics import mean
import json

# SELECT NO RANDOM STATE
# torch.manual_seed(0)
# np.random.seed(0)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def train_dnn(model, dataset, data_path, metrics_save_path, key, mdl_save_path, max_epoch = 200, early_stopping_patience = 10, filt_path = None):
    
    n_subjects, n_chan, batch_size = get_params(dataset)

    for n in range(n_subjects):
        
        subject = get_subject(n, n_subjects)
        
        if dataset == 'jaulab':
            n_chan = check_jaulab_chan(subject)

        print(f'Training {model} with {dataset} data on {subject}...')

        if model == 'FCNN':
            mdl = FCNN(n_hidden = 3, dropout_rate=0.45, n_chan=n_chan)
            optimizer = torch.optim.NAdam(mdl.parameters(), lr=1e-6, weight_decay = 1e-4)
        else:
            mdl = CNN(F1=8, D=8, F2=64, dropout=0.2, input_channels=n_chan)
            optimizer = torch.optim.NAdam(mdl.parameters(), lr=2e-5, weight_decay = 1e-8)

        mdl.to(device)

        train_set, val_set = get_Dataset(dataset, data_path, subject, n, train=True, norm_stim=True, filt=True, filt_path=filt_path)
        train_loader, val_loader = DataLoader(train_set, batch_size, shuffle=False, pin_memory=True),  DataLoader(val_set, batch_size, shuffle=False, pin_memory=True)


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

            # validation
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
            if mean_accuracy > best_accuracy or epoch == 0:
                best_accuracy = mean_accuracy
                best_epoch = epoch
                best_state_dict = mdl.state_dict()

        # save best final model
        mdl_folder = os.path.join(mdl_save_path, dataset + '_data', model+'_'+key)
        if not os.path.exists(mdl_folder):
            os.makedirs(mdl_folder)
        torch.save(
            best_state_dict, 
            os.path.join(mdl_folder, subject+'_'+f'_epoch={epoch}_acc={mean_accuracy:.4f}.ckpt')
        )

        # save corresponding metrics
        val_folder = os.path.join(metrics_save_path, dataset + '_data', 'val', model+'_'+key)
        if not os.path.exists(val_folder):
            os.makedirs(val_folder)
        # save corresponding metrics
        train_folder = os.path.join(metrics_save_path, dataset + '_data', 'train', model+'_'+key)
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        json.dump(train_loss, open(os.path.join(train_folder, subject+'_train_loss'+f'_epoch={epoch}_acc={mean_accuracy:.4f}'),'w'))
        json.dump(val_loss, open(os.path.join(val_folder, subject+'_val_loss'+f'_epoch={epoch}_acc={mean_accuracy:.4f}'),'w'))

def train_ridge(dataset, data_path, mdl_save_path, key, start_lag=0, end_lag=50, original=False, filt_path = None):

    # FOR ALL SUBJECTS
    n_subjects, n_chan, batch_size = get_params(dataset)
    alphas = np.logspace(-7,7, 15)

    for n in range(n_subjects):

        subject = get_subject(n, n_subjects)
        
        mdl = Ridge(start_lag=start_lag, end_lag=end_lag, alpha=alphas, original=original)
        
        train_set, val_set = get_Dataset(dataset, data_path, subject, n, train=True, norm_stim=True, filt=True, filt_path=filt_path)
        
        if dataset == 'fulsang' or dataset == 'jaulab':
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
        model = 'Ridge_'+key if not original else 'Ridge_Original_'+key
        mdl_folder = os.path.join(mdl_save_path, dataset + '_data', model)
        if not os.path.exists(mdl_folder):
            os.makedirs(mdl_folder)
        subj = get_subject(n, n_subjects)
        save_path = os.path.join(mdl_folder, subj+f'_alpha={best_alpha}_acc={scores[best_alpha]:.4f}')

        pickle.dump(mdl, open(save_path, "wb"))

def leave_one_out_ridge(dataset, datapath, window, original, subject, save_path, start_lag = 0, end_lag = 26):

    # Create alpha values from 10^-7 to 10^7
    # alphas = np.logspace(-7,7, 15)

    mdl = Ridge(start_lag=start_lag, end_lag=end_lag, alpha=0.2, original=original)

    # if window > 26:
    #     trials = 48
    # else:
    #     trials = 96

    trials = 60
    
    corr = []
    attended_correct = np.zeros((15))

    for idx in range(trials):

        print(f'Ridge: Leaving trial {idx} out using a {window}s window on subject {subject}')

        # Returns the trials for training and the one for validating on the splitted windows
        data_set = FulsangDatasetWindows(datapath, subject, window, cross_val_index=idx)

        mdl.fit(data_set.eeg.T, data_set.stima[:, np.newaxis])

        # VALIDATE AND SELECT BEST ALPHA
        scores_a = mdl.model_selection(data_set.val_eeg.T, data_set.val_stima[:, np.newaxis])
        scores_b = mdl.model_selection(data_set.val_eeg.T, data_set.val_stimb[:, np.newaxis])
        for alpha_idx in range(len(scores_a)):
            if scores_a[alpha_idx] > scores_b[alpha_idx]:
                attended_correct[alpha_idx] += 1

        best_alpha = mdl.best_alpha_idx

        print(f'Model for subject {subject} trained with a score of {scores_a[best_alpha]} with alpha = {best_alpha}')

        corr.append(scores_a[best_alpha])

    print(f'Mean corr with best alpha: ', mean(corr))
    print(f'Number of correct classification for each alpha: ', attended_correct)



    










