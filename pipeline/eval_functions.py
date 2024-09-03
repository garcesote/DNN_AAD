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

def eval_dnn(model:str, dataset:str, subjects:list, window_len:int, data_path:str, dst_save_path:str, mdl_path:str, 
             accuracy=False, population = False, filt = False, filt_path = None):
    
    """Training parameters
    
    model: str
        introduce the model between 'fcnn', 'cnn'
    
    dataset: str
        introduce the name of the dataset between 'fulsang', 'jaulab', 'hugo'

    subjects: list
        list of subjects you want your network to be evaluated on

    window_len: int
        lenght of the window used for evaluating

    data_path: str
        path where the datasets are located

    matrics_save_path: string
        save path for the train and val loss
    
    mdl_save_path: string
        save path for the trained model

    max_epoch: int
        maximun number of epoch during training

    early_stoping_patience: int
        number of waiting epoch before stop training because not improving loss

    population: bool
        select if you want to train on the subject specific mode or on the population where
        the subject introduced is ignored and the network gets trained on the rest

    filt: bool
        select if you want your eeg signal to be filtered (useful only when selecting fulsang 
        or jaulab data) 

    filt_path: str
        when filt==True the path from where eeg signals get selected
    
    """

    print('Evaluating '+model+' on '+dataset+' dataset')

    n_subjects, n_chan, batch_size, _ = get_params(dataset)

    if not isinstance(subjects, list):
        subjects = [subjects]

    eval_results = {}
    nd_results = {} # contruct a null distribution when evaluating

    # insert the number of samples for performing the circular time shift to obtain the null distribution, in this case between 1 and 2s
    time_shift = 200 if dataset == 'hugo' else 100

    for n, subj in enumerate(subjects):

        if dataset == 'jaulab':
            n_chan = check_jaulab_chan(subj)

        test_set = get_Dataset(dataset, data_path, subj, train=False, acc=accuracy, norm_stim=True, 
                               population=population, filt=filt, filt_path=filt_path)
        test_loader = DataLoader(test_set, window_len, shuffle=False, pin_memory=True)
        
        # OBTAIN MODEL PATH
        filename = dataset
        folder_path = os.path.join(mdl_path , dataset + '_data', model)
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
        corr = []
        nd_corr = []
        with torch.no_grad():
            for i, (x,y) in enumerate(test_loader):
                
                x = x.to(device, dtype=torch.float)
                y = y.to(device, dtype=torch.float)
        
                y_hat, loss = mdl(x)

                nd_acc = correlation(torch.roll(y, time_shift), y_hat)
                acc = correlation(y, y_hat)

                corr.append(acc.item())
                nd_corr.append(nd_acc.item())

        eval_results[subj] = corr
        nd_results[subj] = nd_corr

        print(f'Subject {subj} | corr_mean {mean(corr)}')

    # SAVE RESULTS
    dest_path = os.path.join(dst_save_path, dataset + '_data')
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    filename = model+'_Results'
    json.dump(eval_results, open(os.path.join(dest_path, filename),'w'))
    filename = model+'_nd_Results'
    json.dump(nd_results, open(os.path.join(dest_path, filename),'w'))


def eval_ridge(dataset:str, subjects:list, window_len:int, data_path:str, mdl_path:str, dst_save_path:str, 
               original = False, filt=False, filt_path = None):

    n_subjects, n_chan, batch_size, _= get_params(dataset)
    eval_results = {}
    nd_results = {} # contruct a null distribution for comparing it with the results

    # insert the number of samples for performing the circular time shift to obtain the null distribution, in this case between 1 and 2s
    time_shift = 200 if dataset == 'hugo' else 100

    for n, subj in enumerate(subjects):

        # CARGA EL MODELO
        model = 'Ridge' if not original else 'Ridge_Original'
        mdl_folder_path = os.path.join(mdl_path, dataset + '_data', model)
        filename = get_filname(mdl_folder_path, subj)
        mdl = pickle.load(open(os.path.join(mdl_folder_path, filename), 'rb'))

        # CARGA EL TEST_SET
        test_dataset = get_Dataset(dataset, data_path, subj, train=False, norm_stim=True, filt=True, filt_path=filt_path)
        if dataset == 'fulsang' or dataset == 'jaulab':
            test_eeg, test_stim = test_dataset.eeg, test_dataset.stima
        else:
            test_eeg, test_stim = test_dataset.eeg, test_dataset.stim

        test_stim_nd = torch.roll(torch.tensor(test_stim), time_shift)

        # EVALÚA EN FUNCIÓN DEL MEJOR ALPHA/MODELO OBTENIDO
        scores = mdl.score_in_batches(test_eeg.T, test_stim[:, np.newaxis], batch_size=window_len)
        scores_nd = mdl.score_in_batches(test_eeg.T, test_stim_nd[:, np.newaxis], batch_size=window_len) # ya selecciona el best alpha solo
        eval_results[subj] = [score for score in np.squeeze(scores)]
        nd_results[subj] = [score for score in np.squeeze(scores_nd)]

        print(f'Sujeto {subj} | corr: {np.mean(scores, axis=0)}')

    dest_path = dest_path = os.path.join(dst_save_path, dataset + '_data')
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    filename = 'Ridge_Results' if not original else 'Ridge_Original_Results'
    json.dump(eval_results, open(os.path.join(dest_path, filename),'w'))
    filename = 'Ridge_nd_Results' if not original else 'Ridge_Original_nd_Results'
    json.dump(nd_results, open(os.path.join(dest_path, filename),'w'))


# Save the decoding accuracy of each model, only fulsang and jaulab datasets are valid as hugo_data doesn't present two competing stimuli
def decode_attention(model:str, dataset:str, subjects:list, window_len:int, data_path:str, mdl_path:str, dst_save_path:str, 
                    population = False, filt = True, filt_path = None):

    n_subjects, n_chan, batch_size, _ = get_params(dataset)
    accuracies = []

    if not isinstance(subjects, list):
        subjects = [subjects] 
    

    print(f'Decoding {model} on {dataset} dataset with a window of {str(window_len//64)}s')

    for n, subj in enumerate(subjects):

        if dataset == 'jaulab':
            n_chan = check_jaulab_chan(subj)

        # LOAD DATA
        test_set = get_Dataset(dataset, data_path, subj, train=False, acc=True, norm_stim=True, population=population, filt=filt, filt_path=filt_path)
        test_loader = DataLoader(test_set, window_len, shuffle=False, pin_memory=True)

        attended_correct = 0

        if model == 'Ridge' or  model == 'Ridge_Original':
            
            # CARGA EL MODELO
            mdl_folder_path = os.path.join(mdl_path, dataset + '_data', model)
            filename = get_filname(mdl_folder_path, subj)
            mdl = pickle.load(open(os.path.join(mdl_folder_path, filename), 'rb'))


            test_eeg, test_stima, test_stimb = test_set.eeg, test_set.stima, test_set.stimb
            scores_a = np.squeeze(mdl.score_in_batches(test_eeg.T, test_stima[:, np.newaxis], batch_size=window_len)) # ya selecciona el best alpha solo
            scores_b = np.squeeze(mdl.score_in_batches(test_eeg.T, test_stimb[:, np.newaxis], batch_size=window_len)) # ya selecciona el best alpha solo

            for i in range(len(scores_a)):
                score_a = scores_a[i]
                score_b = scores_b[i]

                if score_a > score_b:
                    attended_correct += 1

            dec_accuracy = (attended_correct / len(scores_a)) * 100
        
        else:

            # OBTAIN MODEL PATH
            folder_path = os.path.join(mdl_path , dataset + '_data', model)
            filename = get_filname(folder_path, subj)
        
            model_path = os.path.join(folder_path, filename)

            # LOAD THE MODEL
            if model=='CNN':
                mdl = CNN(F1=8, D=8, F2=64, dropout=0.2, input_channels=n_chan)
            else:
                mdl = FCNN(n_hidden = 3, dropout_rate=0.45, n_chan=n_chan)

            mdl.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
            mdl.to(device)

            with torch.no_grad():
                for i, (eeg, stima, stimb) in enumerate(test_loader):
                    
                    eeg = eeg.to(device, dtype=torch.float)
                    stima = stima.to(device, dtype=torch.float)
                    stimb = stimb.to(device, dtype=torch.float)
            
                    preds, loss = mdl(eeg)

                    acc_a = correlation(stima, preds)
                    acc_b = correlation(stimb, preds)

                    if acc_a > acc_b:
                        attended_correct +=1

            dec_accuracy = (attended_correct/len(test_loader)) *100

        accuracies.append(dec_accuracy)
        print(f'Subject: {subj} | acc: {dec_accuracy}')

    dest_path = dest_path = os.path.join(dst_save_path, dataset + '_data', model)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    filename = str(window_len)+'_accuracies'

    json.dump(accuracies, open(os.path.join(dest_path, filename),'w'))
                


        
