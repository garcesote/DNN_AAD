import torch
import os
from utils.datasets import HugoMapped, FulsangDataset, JaulabDataset
from utils.dnn import CNN, FCNN
from torch.utils.data import DataLoader
from utils.functional import correlation, get_subject
from statistics import mean
import numpy as np
import pickle
import json

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def get_filname(mdl_folder_path, subject):
    list_dir = os.listdir(mdl_folder_path)
    filename = ''
    for file in list_dir:
        if subject in file:
            if subject == 'S1':
                idx = file.index(subject)
                if file[idx+2] == '_': # si el siguiente caracter al S1 es un barra baja añade al diccionario
                    filename = file
            else:
                filename = file
    return filename

subj_jaulab_60electr = ['S13','S16']

def eval_dnn(model, dataset, data_path, dst_save_path, mdl_path, key):

    print('Evaluating '+model+' on '+dataset+' dataset')

    # FOR ALL SUBJECTS
    data_subj = {'fulsang': 18, 'jaulab': 17, 'hugo': 13}
    n_subjects = data_subj[dataset]
    n_channels = {'fulsang': 64, 'jaulab': 61, 'hugo': 63}
    n_chan = n_channels[dataset]
    batch_sizes = {'fulsang': 128, 'jaulab': 128, 'hugo': 256} # training with windows of 2s
    batch_size = batch_sizes[dataset]

    eval_results = {}

    for n in range(n_subjects):

        subj = get_subject(n, n_subjects)
        if dataset=='jaulab' and subj in subj_jaulab_60electr :
            n_chan = 60
        elif dataset=='jaulab':
            n_chan = 61

        # LOAD DATA
        if dataset == 'fulsang':
            test_set = FulsangDataset(data_path, 'test', subj)
            test_loader = DataLoader(test_set, batch_size = batch_size, pin_memory=True)
        if dataset == 'jaulab':
            test_set = JaulabDataset(data_path, 'test', subj)
            test_loader = DataLoader(test_set, batch_size = batch_size, pin_memory=True)        
        else:
            test_set = HugoMapped(range(12,15), data_path, participant=n)
            test_loader = DataLoader(test_set, batch_size = batch_size, pin_memory=True)

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

    n_subjects = 18 if dataset=='fulsang' or dataset == 'jaulab' else 13
    batch_size = 125 if dataset=='fulsang' or dataset == 'jaulab' else 256
    eval_results = {}

    for n in range(n_subjects):

        # CARGA EL MODELO
        subj = get_subject(n, n_subjects)
        mdl_folder_path = os.path.join(mdl_path, dataset + '_data', 'Ridge_'+key)
        filename = get_filname(mdl_folder_path, subj)
        mdl = pickle.load(open(filename, 'rb'))

        # CARGA EL TEST_SET
        test_dataset = HugoMapped(range(12, 15), data_path, participant=n)
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