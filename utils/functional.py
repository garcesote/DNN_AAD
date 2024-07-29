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

# Returns the corresponding subject data in datasets
def get_Dataset(dataset:str, data_path:str, subject:str, n: int, train = True, acc=False):

    if dataset == 'fulsang':
        train_set = FulsangDataset(data_path, 'train', subject)
        val_set = FulsangDataset(data_path, 'val', subject)
        test_set = FulsangDataset(data_path, 'test', subject) if not acc else FulsangDataset(data_path, 'test', subject, mode='acc')
    elif dataset == 'jaulab':
        train_set = JaulabDataset(data_path, 'train', subject)
        val_set = JaulabDataset(data_path, 'val', subject)
        test_set = JaulabDataset(data_path, 'test', subject) if not acc else JaulabDataset(data_path, 'test', subject, mode='acc')
    else:
        train_set = HugoMapped(range(9), data_path, participant=n)
        val_set = HugoMapped(range(9, 12), data_path, participant=n)
        test_set = HugoMapped(range(12, 15), data_path, participant=n)

    if train:
        return train_set, val_set
    else:
        return test_set