from torch.utils.data import Dataset
import scipy
import torch
import numpy as np
import h5py
import os

def get_windows(eeg_data, stima_data, stimb_data, n_window, window_samples, len_trial):

     # If the number is less than 1 (only case of win_size=50s) concat the trials 2 by 2
    if n_window < 1:
        n_window = 1
        eeg_windows = [normalize(torch.tensor(eeg_data[trial]+eeg_data[trial+1]).T) for trial in range(0, len_trial, 2)]
        stima_windows = [torch.squeeze(torch.tensor(stima_data[trial]+stima_data[trial+1])) for trial in range(0, len_trial, 2)]
        stimb_windows = [torch.squeeze(torch.tensor(stimb_data[trial]+stima_data[trial+1])) for trial in range(0, len_trial, 2)]
    
    # If not return the necessary windows (eg: 26s returns a complete trial, 10s returns 2 windows for trial)
    else:
        eeg_windows = [normalize(torch.tensor(eeg_data[trial][win*window_samples:(win+1)*window_samples,:]).T) for win in range(int(n_window)) for trial in range(0, len_trial)]
        stima_windows = [torch.squeeze(torch.tensor(stima_data[trial][win*window_samples:(win+1)*window_samples,:])) for win in range(int(n_window)) for trial in range(0, len_trial)]
        stimb_windows = [torch.squeeze(torch.tensor(stimb_data[trial][win*window_samples:(win+1)*window_samples,:])) for win in range(int(n_window)) for trial in range(0, len_trial)]
        
    return eeg_windows, stima_windows, stimb_windows

# turn a tensor to 0 mean and std of 1 with shape (C, T) and return shape (C)   
def normalize(tensor: torch.tensor):

    # unsqueeze necesario para el broadcasting (10) => (10, 1)
    mean = (torch.mean(tensor, dim=1)).unsqueeze(1)
    std = torch.std(tensor, dim=1).unsqueeze(1)

    return (tensor - mean) / std

# turn a tensor to 0 mean and std of 1 with shape (T)
def normalize_stim(tensor: torch.tensor):

    mean = torch.mean(tensor)
    std = torch.std(tensor)

    return (tensor - mean) / std

class HugoMapped(Dataset):
    '''
    dataloader for reading Hugo's data
    '''
    def __init__(self, parts_list, data_dir, participant=0, num_input=50, channels = np.arange(63)):

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

class FulsangDataset(Dataset):

    def __init__(self, folder_path, trials, subjects, window = 50, acc=False, norm_stim=False, filt = False, filt_path=None):

        if not isinstance(subjects, list):
            subjects = [subjects]
            
        eeg = []
        stima = []
        stimb = []

        for subject in subjects:
            data_path = os.path.join(folder_path ,subject + '_data_preproc.mat')
            preproc_data = scipy.io.loadmat(data_path)

            # Array con n trials y dentro las muestras de audio y eeg
            stima_data = preproc_data['data']['wavA'][0,0][0,trials]
            stimb_data = preproc_data['data']['wavB'][0,0][0,trials]
            n_trial = len(trials)

            if filt:
                eeg_data = np.load(os.path.join(filt_path, subject+'_data_filt.npy'), allow_pickle=True)[trials]
            else:
                eeg_data = preproc_data['data']['eeg'][0,0][0,trials]

            # si hay mas canales de la cuenta selecciono los 64 primeros
            if eeg_data[0].shape[1] > 64:
                eeg_data = [trial[:,:64] for trial in eeg_data]
        
            # Concatenar en un tensor todas os trials del sujeto (muestras * trials, canales) => (T * N, C).T => (C, T * N)
            eeg.append(torch.hstack([normalize(torch.tensor(eeg_data[trial]).T) for trial in range(n_trial)]))
            stima.append(torch.squeeze(torch.vstack([torch.tensor(stima_data[trial]) for trial in range(n_trial)])) if not norm_stim else torch.squeeze(normalize_stim(torch.vstack([torch.tensor(stima_data[trial]) for trial in range(n_trial)]))))
            stimb.append(torch.squeeze(torch.vstack([torch.tensor(stimb_data[trial]) for trial in range(n_trial)])) if not norm_stim else torch.squeeze(normalize_stim(torch.vstack([torch.tensor(stimb_data[trial]) for trial in range(n_trial)]))))

        # Concateno en un tensor global la información de los sujetos indicados
        self.eeg = torch.hstack(eeg)
        self.stima = torch.cat(stima)
        self.stimb = torch.cat(stimb)

        self.trials = n_trial
        self.samples = eeg_data[0].shape[0]
        self.channels = eeg_data[0].shape[1]
        self.window = window
        self.subject = subjects
        self.n_subjects = len(subjects)
        self.acc = acc

    def __getitem__(self,idx):

        rest = self.window - (self.samples * self.trials * self.n_subjects - idx)

        if self.acc:
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
                window = self.eeg[:, idx:idx+self.window]
                return window, self.stima[idx]
            # Si llega al final, añadirle las muestras que faltan
            else:
                window = torch.hstack([self.eeg[:, idx:idx+self.window] , self.eeg[:, 0:rest]])
                return window, self.stima[idx]
            
    def __len__(self):
        return self.samples * self.trials
    
class JaulabDataset(Dataset):

    def __init__(self, folder_path, trials, subjects, window = 50, acc=False, norm_stim = False, filt = False, filt_path=None):
        
        if not isinstance(subjects, list):
            subjects = [subjects]

        eeg = []
        stima = []
        stimb = []

        for subject in subjects:
            data_path = os.path.join(folder_path ,subject + '_preproc.mat')
            preproc_data = scipy.io.loadmat(data_path)

            # Array con n trials y dentro las muestras de audio y eeg
            stima_data = preproc_data['data']['wavA'][0,0][0,trials]
            stimb_data = preproc_data['data']['wavB'][0,0][0,trials]
            len_trial = len(trials)

            if filt:
                eeg_data = np.load(os.path.join(filt_path, subject+'_data_filt.npy'), allow_pickle=True)[trials]
            else:
                eeg_data = preproc_data['data']['eeg'][0,0][0,trials]
        
            # Concatenar en un tensor todas os trials del sujeto (muestras * trials, canales) => (T * N, C).T => (C, T * N)
            eeg.append(torch.hstack([normalize(torch.tensor(eeg_data[trial]).T) for trial in range(len_trial)]))
            stima.append(torch.squeeze(torch.vstack([torch.tensor(stima_data[trial]) for trial in range(len_trial)])) if not norm_stim else torch.squeeze(normalize_stim(torch.vstack([torch.tensor(stima_data[trial]) for trial in range(len_trial)]))))
            stimb.append(torch.squeeze(torch.vstack([torch.tensor(stimb_data[trial]) for trial in range(len_trial)])) if not norm_stim else torch.squeeze(normalize_stim(torch.vstack([torch.tensor(stimb_data[trial]) for trial in range(len_trial)]))))

        # Concateno en un tensor global la información de los sujetos indicados
        self.eeg = torch.hstack(eeg)
        self.stima = torch.cat(stima) 
        self.stimb = torch.cat(stimb)

        self.trials = len_trial
        self.samples = eeg_data[0].shape[0]
        self.channels = eeg_data[0].shape[1]
        self.window = window
        self.subject = subjects
        self.n_subjects = len(subjects)
        self.acc = acc

    def __getitem__(self,idx):
        rest = self.window - (self.samples * self.trials  * self.n_subjects - idx)

        if self.acc:
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
    

class JaulabDatasetWindows(Dataset):

    def __init__(self, folder_path, subject, window = 50, cross_val_index = 0):

        data_path = os.path.join(folder_path ,subject + '_preproc.mat')
        preproc_data = scipy.io.loadmat(data_path)

        n_trials = 96
        trials = np.arange(n_trials)

        # Array con n trials y dentro las muestras de audio y eeg
        eeg_data = preproc_data['data']['eeg'][0,0][0,trials]
        stima_data = preproc_data['data']['wavA'][0,0][0,trials]
        stimb_data = preproc_data['data']['wavB'][0,0][0,trials]
        len_trial = len(trials)

        self.trials = len_trial
        self.samples = eeg_data[0].shape[0]
        self.channels = eeg_data[0].shape[1]
        self.subject = subject
        self.fs = 64
        self.window_samples = self.fs * window

        # Get the number of windows per trial
        self.n_window = self.samples // self.window_samples

        # Get the data splitted by windows (list) with the corresponding data of that window (tensors)
        eeg_windows, stima_windows, stimb_windows = get_windows(eeg_data, stima_data, stimb_data, self.n_window, self.window_samples, len_trial)

        self.windows = len(eeg_windows)

        # Returns the concatenated window data outside from the trial selected for validation 
        self.eeg = torch.hstack([eeg[:, :self.window_samples] for i, eeg in enumerate(eeg_windows) if i not in list(range(cross_val_index, cross_val_index + self.n_window))])
        self.stima = torch.cat([stima[:self.window_samples] for i, stima in enumerate(stima_windows) if i not in list(range(cross_val_index, cross_val_index + self.n_window))])
        self.stimb = torch.cat([stimb[:self.window_samples] for i, stimb in enumerate(stimb_windows) if i not in list(range(cross_val_index, cross_val_index + self.n_window))])

        # Returns the concatenated window data corresponding to the validation trial
        self.val_eeg = torch.hstack([eeg[:, :self.window_samples] for eeg in eeg_windows[cross_val_index: cross_val_index + self.n_window]])
        self.val_stima = torch.cat([stima[:self.window_samples] for stima in stima_windows[cross_val_index: cross_val_index + self.n_window]])
        self.val_stimb = torch.cat([stimb[:self.window_samples] for stimb in stimb_windows[cross_val_index: cross_val_index + self.n_window]])

class FulsangDatasetWindows(Dataset):

    def __init__(self, folder_path, subject, window = 50, cross_val_index = 0):

        data_path = os.path.join(folder_path ,subject + '_data_preproc.mat')
        preproc_data = scipy.io.loadmat(data_path)

        n_trials = 60
        trials = np.arange(n_trials)

        # Array con n trials y dentro las muestras de audio y eeg
        eeg_data = preproc_data['data']['eeg'][0,0][0,trials]
        stima_data = preproc_data['data']['wavA'][0,0][0,trials]
        stimb_data = preproc_data['data']['wavB'][0,0][0,trials]
        len_trial = len(trials)

        # si hay mas canales de la cuenta selecciono los 64 primeros
        if eeg_data[0].shape[1] > 64:
            eeg_data = [trial[:,:64] for trial in eeg_data]

        self.trials = len_trial
        self.samples = eeg_data[0].shape[0]
        self.channels = eeg_data[0].shape[1]
        self.subject = subject
        self.fs = 64
        self.window_samples = self.fs * window

        # Get the number of windows per trial
        self.n_window = self.samples // self.window_samples

        # Get the data splitted by windows (list) with the corresponding data of that window (tensors)
        eeg_windows, stima_windows, stimb_windows = get_windows(eeg_data, stima_data, stimb_data, self.n_window, self.window_samples, len_trial)

        self.windows = len(eeg_windows)

        # Returns the concatenated window data outside from the trial selected for validation 
        self.eeg = torch.hstack([eeg[:, :self.window_samples] for i, eeg in enumerate(eeg_windows) if i not in list(range(cross_val_index, cross_val_index + self.n_window))])
        self.stima = torch.cat([stima[:self.window_samples] for i, stima in enumerate(stima_windows) if i not in list(range(cross_val_index, cross_val_index + self.n_window))])
        self.stimb = torch.cat([stimb[:self.window_samples] for i, stimb in enumerate(stimb_windows) if i not in list(range(cross_val_index, cross_val_index + self.n_window))])

        # Returns the concatenated window data corresponding to the validation trial
        self.val_eeg = torch.hstack([eeg[:, :self.window_samples] for eeg in eeg_windows[cross_val_index: cross_val_index + self.n_window]])
        self.val_stima = torch.cat([stima[:self.window_samples] for stima in stima_windows[cross_val_index: cross_val_index + self.n_window]])
        self.val_stimb = torch.cat([stimb[:self.window_samples] for stimb in stimb_windows[cross_val_index: cross_val_index + self.n_window]])