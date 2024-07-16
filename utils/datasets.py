from torch.utils.data import Dataset
import scipy
from utils.functional import get_trials, normalize
import torch
import numpy as np
import h5py
import os


class FulsangDataset(Dataset):

    def __init__(self, folder_path, split, subject, window = 50, mode='corr'):

        data_path = os.path.join(folder_path ,subject + '_data_preproc.mat')
        preproc_data = scipy.io.loadmat(data_path)

        trials = get_trials(split)

        # Array con n trials y dentro las muestras de audio y eeg
        eeg_data = preproc_data['data']['eeg'][0,0][0,trials]
        stima_data = preproc_data['data']['wavA'][0,0][0,trials]
        stimb_data = preproc_data['data']['wavB'][0,0][0,trials]
        len_trial = len(trials)

        # si hay mas canales de la cuenta selecciono los 64 primeros
        if eeg_data[0].shape[1] > 64:
            eeg_data = [trial[:,:64] for trial in eeg_data]

        # Concatenar en un tensor todas las muestras (muestras * trials, canales) => (T * N, C).T => (C, T * N)
        self.eeg = torch.hstack([normalize(torch.tensor(eeg_data[trial]).T) for trial in range(len_trial)])
        self.stima = torch.squeeze(torch.vstack([torch.tensor(stima_data[trial]) for trial in range(len_trial)]))
        self.stimb = torch.squeeze(torch.vstack([torch.tensor(stimb_data[trial]) for trial in range(len_trial)]))

        self.trials = len_trial
        self.samples = eeg_data[0].shape[0]
        self.channels = eeg_data[0].shape[1]
        self.window = window
        self.subject = subject
        self.mode = mode

    def __getitem__(self,idx):
        rest = self.window - (self.samples * self.trials - idx)

        if self.mode == 'acc':
            # Si se coge la ventana entera sin llegar al final
            if rest < 0:
                return self.eeg[:, idx:idx+self.window], self.stima[idx], self.stimb[idx]
            # Si llega al final, añadirle las muestras que faltan
            else:
                window = torch.hstack([self.eeg[:, idx:idx+self.window] , self.eeg[:, 0:rest]])
                return window, self.stima[idx], self.stimb[idx]
        else:
            # Si se coge la ventana entera sin llegar al final
            if rest < 0:
                return self.eeg[:, idx:idx+self.window], self.stima[idx]
            # Si llega al final, añadirle las muestras que faltan
            else:
                window = torch.hstack([self.eeg[:, idx:idx+self.window] , self.eeg[:, 0:rest]])
                return window, self.stima[idx]
    def __len__(self):
        return self.samples * self.trials
    
class HugoMapped(Dataset):
    '''
    dataloader for reading Hugo's data
    '''
    def __init__(
        self,
        parts_list,
        data_dir,
        participant=0,
        num_input=50,
        channels = np.arange(63)):

        self.parts_list = parts_list
        self.data_dir=data_dir
        self.num_input=num_input
        self.channels=channels

        if type(participant)==type(int()):
            self.participants=[participant]
        else:
            self.participants=participant

        self._initialise_data() 

    def _initialise_data(self):

        eeg = []
        stim = []

        with h5py.File(self.data_dir, "r") as f:
            
            for each_participant in self.participants:
                eeg += [f['eeg/P0{}/part{}/'.format(each_participant, j)][:][self.channels] for j in self.parts_list]
                stim += [f['stim/part{}/'.format(j)][:] for j in self.parts_list]

        self.eeg = np.hstack(eeg)
        self.stim = np.hstack(stim)

    def __getitem__(self, idx):

        return self.eeg[:, idx:idx+self.num_input], self.stim[idx]
    
    def __len__(self):
        return self.stim.size - self.num_input