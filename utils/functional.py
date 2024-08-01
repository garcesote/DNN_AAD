import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from utils.datasets import FulsangDataset, HugoMapped, JaulabDataset

# Returns the name of the subject by introducing the index
def get_subject(idx: int, n_subjects: int):
    subjects = ['S' + str(i+1) for i in range(n_subjects)]
    return subjects[idx]

# Return the required parameters depending on the different dataset
def get_params(dataset: str):

    data_subj = {'fulsang': 18, 'jaulab': 17, 'hugo': 13}
    n_subjects = data_subj[dataset]
    n_channels = {'fulsang': 64, 'jaulab': 61, 'hugo': 63}
    n_chan = n_channels[dataset]
    batch_sizes = {'fulsang': 128, 'jaulab': 128, 'hugo': 256} # training with windows of 2s
    batch_size = batch_sizes[dataset]

    return n_subjects, n_chan, batch_size

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
def get_excluded_subject(subjects):
    n_subjects = len(subjects) + 1
    all_subjects = ['S'+str(n) for n in range(1, n_subjects)]
    subject = list(set(all_subjects) - set(subjects))[0]
    return subject

# Returns the corresponding subject data in dataset
def get_Dataset(dataset:str, data_path:str, subjects: list, train = True, acc=False, norm_stim=False, filt=False, filt_path=None):

    '''
    Input params:
        dataset: select dataset between 'fulsang', 'jaulab' or 'hugo
        data_path: path where the data from the subjects is located
        subject: specify the subject or subjects from which get the data, eg: ['S1'] or ['S2' ... 'S18']
        n: index of the corresponding subject
        train: select whether you are getting val and train sets or the test set
        acc: returns the dataset with the attended only or attended and unattended stim for decode the accuracy
        norm_stim : normalize the stimulus of fulsang and jaulab dataset
        file: select the filtered fulsang or jaulab data
        filt_path: select the path of the filtered data
    '''

    if not isinstance(subjects, list):
        subjects = [subjects]

    n = [int(subj[1:]) for subj in subjects]

    if len(subjects) > 1:
        val_test_subj = get_excluded_subject(subjects)
    else:
        val_test_subj = subjects

    if dataset == 'fulsang':
        if train:
            train_set = FulsangDataset(data_path, 'train', subjects, norm_stim=norm_stim, filt=filt, filt_path=filt_path)
            val_set = FulsangDataset(data_path, 'val', subjects, norm_stim=norm_stim, filt=filt, filt_path=filt_path)
        else:
            test_set = FulsangDataset(data_path, 'test', subjects, acc=acc,  norm_stim=norm_stim, filt=filt, filt_path=filt_path)
    elif dataset == 'jaulab':
        if train:
            train_set = JaulabDataset(data_path, 'train', subjects, norm_stim=norm_stim, filt=filt, filt_path=filt_path)
            val_set = JaulabDataset(data_path, 'val', subjects,  norm_stim=norm_stim, filt=filt, filt_path=filt_path)
        else:
            test_set = JaulabDataset(data_path, 'test', subjects, acc=acc, norm_stim=norm_stim, filt=filt, filt_path=filt_path)
    else:
        if train:
            train_set = HugoMapped(range(9), data_path, participant=n)
            val_set = HugoMapped(range(9, 12), data_path, participant=n)
        else:
            test_set = HugoMapped(range(12, 15), data_path, participant=n)

    if train:
        return train_set, val_set
    else:
        return test_set