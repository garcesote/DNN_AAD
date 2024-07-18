import torch
import os
from utils.datasets import HugoMapped, FulsangDataset, JaulabDataset
from utils.dnn import CNN, FCNN
from torch.utils.data import DataLoader
from utils.functional import correlation, get_subject, check_jaulab_chan, get_params, get_filname, get_Dataset
from statistics import mean
import numpy as np
import pickle
import json

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def eval_dnn(model, dataset, data_path, dst_save_path, mdl_path, key, accuracy=False):

    print('Evaluating '+model+' on '+dataset+' dataset')

    # FOR ALL SUBJECTS
    n_subjects, n_chan, batch_size = get_params(dataset)

    eval_results = {}

    for n in range(n_subjects):

        subj = get_subject(n, n_subjects)
        if dataset == 'jaulab':
            n_chan = check_jaulab_chan(subj)

        # LOAD DATA
        test_set = get_Dataset(dataset, data_path, subj, n, train=False)
        test_loader = DataLoader(test_set, batch_size, shuffle=False, pin_memory=True)

        # OBTAIN MODEL PATH
        filename = dataset
        folder_path = os.path.join(mdl_path , dataset + '_data', model+'_'+key)
        filename = get_filname(folder_path, subj)
    
        model_path = os.path.join(folder_path, filename)

        # LOAD THE MODEL
        if model=='CNN':
            mdl = CNN(F1=8, D=8, F2=64, dropout=0.2, input_channels=n_chan)
        else:
            mdl = FCNN(n_hidden = 3, dropout_rate=0.45, n_chan=n_chan)

        mdl.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        mdl.to(device)

        # EVALUATE THE MODEL
        accuracies = []
        with torch.no_grad():
            for i, (x,y) in enumerate(test_loader):
                
                x = x.to(device, dtype=torch.float)
                y = y.to(device, dtype=torch.float)
        
                y_hat, loss = mdl(x)
                acc = correlation(y, y_hat)
                accuracies.append(acc.item())

        eval_results[subj] = accuracies
        print(f'Subject {subj} | acc_mean {mean(accuracies)}')

    # SAVE RESULTS
    dest_path = os.path.join(dst_save_path, dataset + '_data')
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    filename = model+'_'+key+'_Results'
    json.dump(eval_results, open(os.path.join(dest_path, filename),'w'))


def eval_ridge(dataset, data_path, mdl_path, key, dst_save_path):

    n_subjects, n_chan, batch_size= get_params(dataset)
    eval_results = {}

    for n in range(n_subjects):

        # CARGA EL MODELO
        subj = get_subject(n, n_subjects)
        mdl_folder_path = os.path.join(mdl_path, dataset + '_data', 'Ridge_'+key)
        filename = get_filname(mdl_folder_path, subj)
        mdl = pickle.load(open(filename, 'rb'))

        # CARGA EL TEST_SET
        test_dataset = get_Dataset(dataset, data_path, subj, n, train=False)
        test_eeg, test_stim = test_dataset.eeg, test_dataset.stim

        # EVALÚA EN FUNCIÓN DEL MEJOR ALPHA/MODELO OBTENIDO
        scores = mdl.score_in_batches(test_eeg.T, test_stim[:, np.newaxis], batch_size=batch_size) # ya selecciona el best alpha solo
        eval_results[subj] = [score for score in np.squeeze(scores)]

        print(f'Sujeto {n} | accuracy: {np.mean(scores, axis=0)}')

    dest_path = dest_path = os.path.join(dst_save_path, dataset + '_data')
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    filename = 'Ridge_'+key+'_Results'
    json.dump(eval_results, open(os.path.join(dest_path, filename),'w'))