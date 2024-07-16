import torch
import os
from utils.datasets import HugoMapped, FulsangDataset
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

def eval_dnn(model, dataset, data_path, dst_save_path, mdl_path):

    # FOR ALL SUBJECTS
    n_subjects = 18 if dataset=='Fulsang' else 13
    n_chan = 64 if dataset=='Fulsang' else 63
    batch_size = 125 if dataset=='Fulsang' else 256

    eval_results = {}
    mode = model

    def get_filename(subj, file):
        filename = ''
        
        if subj == 'S1':
            idx = file.index(subj)
            if file[idx+2] == '_': # si el siguiente caracter al S1 es un barra baja añade al diccionario
                filename = file
        else:
            filename = file

        return filename

    for n in range(n_subjects):

        subj = get_subject(n, n_subjects)
        
        # LOAD DATA
        if dataset == 'Fulsang':
            test_set = FulsangDataset(data_path, 'test', subj)
            test_loader = DataLoader(test_set, batch_size = batch_size, pin_memory=True)
        else:
            test_set = HugoMapped(range(12,15), data_path, participant=n)
            test_loader = DataLoader(test_set, batch_size = batch_size, pin_memory=True)

        # LOAD MODEL
        filename = dataset.lower() + '_64Hz' if dataset == 'Fulsang' else dataset
        folder_path = os.path.join(mdl_path , dataset.lower() + '_data/')
        list_dir = os.listdir(folder_path)
        filename = ''
        for file in list_dir:
            if mode == 'CNN':
                if subj in file and 'FCNN' not in file:
                    filename = get_filename(subj, file)
            else:
                if subj in file and 'FCNN' in file:
                    filename = get_filename(subj, file)
        model_path = folder_path + filename
        print(subj)
        print(model_path)
        if mode=='CNN':
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
        print(f'Subject {subj} | acc_mean{mean(accuracies)}')

    if dataset == 'Fulsang':
        dest_path = dst_save_path + '/fulsang_64Hz_data'
    else:
        dest_path = dst_save_path + '/hugo_data'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    filename = model+'_Results'
    json.dump(eval_results, open(os.path.join(dest_path, filename),'w'))

def eval_ridge(dataset, data_path, mdl_path, date, dst_save_path):

    n_subjects = 18 if dataset=='Fulsang' else 13
    batch_size = 125 if dataset=='Fulsang' else 256
    eval_results = {}

    for n in range(n_subjects):

        # CARGA EL MODELO
        subj = get_subject(n, n_subjects)
        model_path = mdl_path + '/ridge/ridgeOriginal_'+subj+'_'+date+'_'+dataset
        mdl = pickle.load(open(model_path, 'rb'))

        # CARGA EL TEST_SET
        test_dataset = HugoMapped(range(12, 15), data_path, participant=n)
        test_eeg, test_stim = test_dataset.eeg, test_dataset.stim

        # EVALÚA EN FUNCIÓN DEL MEJOR ALPHA/MODELO OBTENIDO
        scores = mdl.score_in_batches(test_eeg.T, test_stim[:, np.newaxis], batch_size=batch_size) # ya selecciona el best alpha solo
        eval_results[subj] = [score for score in np.squeeze(scores)]

        print(f'Sujeto {n} | accuracy: {np.mean(scores, axis=0)}')

    dest_path = dst_save_path + '/hugo_data' if dataset == 'hugo' else dst_save_path + '/fulsang_64Hz_data'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    filename = 'Ridge_Results'
    json.dump(eval_results, open(os.path.join(dest_path, filename),'w'))