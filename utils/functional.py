import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from utils.datasets import FulsangDataset, HugoMapped, JaulabDataset

# Returns the name of the subject by introducing the index
def get_subject(idx: int, n_subjects: int):
    subjects = ['S' + str(i+1) for i in range(n_subjects)]
    return subjects[idx]

# Return the required trials for splitting correctly the dataset introducing the set
def get_trials(split: str, n_trials: int):

    partitions = [0.7, 0.15, 0.15] # sum to 1
    n_train = int(partitions[0]*n_trials)
    n_val = int(partitions[1]*n_trials)
    n_test = int(partitions[2]*n_trials)

    adjustment = n_trials - (n_train + n_val + n_test)
    n_train += adjustment

    if split == 'train':
        return np.arange(0, n_train)
    elif split == 'val':
        return np.arange(n_train , n_trials - n_val)
    elif split == 'test':
        return np.arange(n_trials - n_val, n_trials)
    else:
        raise ValueError('Field split must be a train/val/test value')
    
# Returns the corresponding trials for the population setting 
def get_population_trials(split: str, n_trials: int):
    # Train case with all the trials corresponding to the train subjects
    if split == 'train':
        return np.arange(n_trials)
    # Val and test case with half of the samples each for the excluded subject
    elif split == 'val':
        return np.arange(0, int(n_trials/2))
    elif split == 'test':
        return np.arange(int(n_trials/2), n_trials)
    else:
        raise ValueError('Field split must be a train/val/test value')

# Return the required parameters depending on the different dataset
def get_params(dataset: str):

    data_subj = {'fulsang': 18, 'jaulab': 17, 'hugo': 13}
    n_subjects = data_subj[dataset]
    n_channels = {'fulsang': 64, 'jaulab': 61, 'hugo': 63}
    n_chan = n_channels[dataset]
    batch_sizes = {'fulsang': 128, 'jaulab': 128, 'hugo': 256} # training with windows of 2s
    batch_size = batch_sizes[dataset]
    n_trials = {'fulsang': 60, 'jaulab': 96, 'hugo': 15} # training with windows of 2s
    n_trials = n_trials[dataset]

    return n_subjects, n_chan, batch_size, n_trials

# Check wether the subject has 60 or 61 electrodes on Jaulab dataset
def check_jaulab_chan(subj: str):
    subj_60_chan = ['S13', 'S16']
    return 60 if subj in subj_60_chan else 61

# Returns the filename related to the subject solving the problem of S1
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

# Calculates the pearson correlation between two tensors
def correlation(x: torch.tensor, y: torch.tensor, eps=1e-8):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)
    return corr

# Returns the subjects not present in the list
def get_other_subjects(subject, dataset):
    
    # Obtain the remaining subject on the population setting for saving the results
    # BUG: in the case of the dataset jaulab, subjects 13 and 16 aren't used because of the electrodes used
    # so delete them from the excludad subjects list adn differ it when defining the number of total subjects
    jaulab_bug_subj = [13, 16]
    ds_subjects = {'fulsang': ['S'+str(n) for n in range(1, 19)], 
                   'jaulab': ['S'+str(n) for n in range(1, 18) if n not in jaulab_bug_subj],
                   'hugo': ['S'+str(n) for n in range(1, 13)]}
    other_subjects = list(set(ds_subjects[dataset]) - set(subject))
    return other_subjects

# Returns the corresponding subject data in dataset
def get_Dataset(dataset:str, data_path:str, subjects: list, train = True, acc=False, 
                norm_stim=False, filt=False, filt_path=None, population = False):

    '''
    Input params:
        dataset: select dataset between 'fulsang', 'jaulab' or 'hugo
        data_path: path where the data from the subjects is located
        subject: specify the subject or subjects from which get the data, eg: ['S1'] or ['S2' ... 'S18']
        train: select whether you are getting val and train sets or the test set
        acc: returns the dataset with the attended only or attended and unattended stim for decode the accuracy
        norm_stim : normalize the stimulus of fulsang and jaulab dataset
        file: select the filtered fulsang or jaulab data
        filt_path: select the path of the filtered data
        population: returns the sets from the population model where half of the trials for the excluded subject are used for test and val and
                    the other subjects used for training, or the set for subject specific with 70%/15%/15% spliy on the specified subject
    '''

    if not isinstance(subjects, list):
        subjects = [subjects]

    n = [int(subj[1:]) for subj in subjects]

    # when population setting select the rest of the subjects for training
    train_subj = subjects if not population else get_other_subjects(subjects, dataset)
    n_train = [int(subj[1:]) for subj in train_subj]
    _, _, _, n_trials = get_params(dataset)

    if dataset == 'fulsang':
        if train:
            train_trials = get_population_trials('train', n_trials) if population else get_trials('train', n_trials)
            train_set = FulsangDataset(data_path, train_trials, train_subj, norm_stim=norm_stim, filt=filt, filt_path=filt_path)
            val_trials = get_population_trials('val', n_trials) if population else get_trials('val', n_trials)
            val_set = FulsangDataset(data_path, val_trials, subjects, norm_stim=norm_stim, filt=filt, filt_path=filt_path)
        else:
            test_trials = get_population_trials('test', n_trials) if population else get_trials('test', n_trials)
            test_set = FulsangDataset(data_path, test_trials, subjects, acc=acc,  norm_stim=norm_stim, filt=filt, filt_path=filt_path)
    elif dataset == 'jaulab':
        if train:
            train_trials = get_population_trials('train', n_trials) if population else get_trials('train', n_trials)
            train_set = JaulabDataset(data_path, train_trials, train_subj, norm_stim=norm_stim, filt=filt, filt_path=filt_path)
            val_trials = get_population_trials('val', n_trials) if population else get_trials('val', n_trials)
            val_set = JaulabDataset(data_path, val_trials, subjects,  norm_stim=norm_stim, filt=filt, filt_path=filt_path)
        else:
            test_trials = get_population_trials('test', n_trials) if population else get_trials('test', n_trials)
            test_set = JaulabDataset(data_path, test_trials, subjects, acc=acc, norm_stim=norm_stim, filt=filt, filt_path=filt_path)
    else:
        if train:
            train_set = HugoMapped(range(9), data_path, participant=n_train)
            val_set = HugoMapped(range(9, 12), data_path, participant=n)
        else:
            test_set = HugoMapped(range(12, 15), data_path, participant=n)

    if train:
        return train_set, val_set
    else:
        return test_set