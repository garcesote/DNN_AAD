import os
import scipy
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.functional import get_other_subjects, get_trials, get_population_trials
import gc 

# turn a tensor to 0 mean and std of 1 with shape (C, T) and return shape (C)   
def normalize_eeg(tensor: torch.tensor):

    # unsqueeze necesario para el broadcasting (C) => (C, 1)
    mean = (torch.mean(tensor, dim=1)).unsqueeze(1)
    std = torch.std(tensor, dim=1).unsqueeze(1)

    return (tensor - mean) / std

# turn a tensor to 0 mean and std of 1 with shape (T)
def normalize_stim(tensor: torch.tensor):

    mean = torch.mean(tensor)
    std = torch.std(tensor)

    return (tensor - mean) / std

# introduce subject index like 'S1' and return index like 'sub-001'
def get_SKL_subj_idx(subject):
    idx = subject[1:]
    zeros = 3 - len(idx) # number of zeros you have to add to the idx
    subj_idx = 'sub-' + ''.join(['0' for _ in range(zeros)] + [idx])
    return subj_idx

class CustomDataset(Dataset):

    """ CustomDataset 
    
    Parameters
    ------------
    
    dataset:str
        intrduce a valid dataset between skl, fulsang, jaulab

    data_path:str
        path for gather the data of the dataset

    split:str
        select the split of the data between 'train', 'test', 'val'

    subjects: list, str
        select the subject / subjects you want your for your data

    window: int
        length of the window used on get_item method

    acc: bool
        return or not the attended stim (only on fulsang or jaulab)
    
    norm_stim: bool
        use normalized stim or not (only on fulsang or jaulab)

    filt: bool
        select if you want to use filtered eeg

    filt_path: str
        path where filtered eeg is located

    population: bool
        select if you want population mode or not
        This parameter change the whole dataset!!:
            - when True: the specified subject or subjects introduced correspond with the leaved out for training data, 
            also different trials in split (100%(rest),50%(subj),50%(subj))
            - when False: the specified subject/subjects used for all splits and the network gets trained on the specified splits (80%,10%,10%)

    __len__()
    -------
    Returns the numeber samples of the whole dataset minus the length of the window for not getting
    out of range

    __getitem__()
    -------
    Returns 'window' samples of eeg separated hop samples by the next one and also the first sample
    of the attended stim.
    if acc is True then it also returns the unattended stimulus first sample
    """

    def __init__(self, dataset, data_path, split, subjects, window, hop, acc=False, 
                 norm_stim=False, filt = False, filt_path=None, population = False,
                 fixed=False, rnd_trials=False):

        if not isinstance(subjects, list):
            subjects = [subjects]

        if not population:
            self.subjects = subjects
        else:
            # When training on population mode all subjects except the specified ore used for training
            if split=='train':
                self.subjects = get_other_subjects(subjects, dataset)
            # For val and test the specified subject is intraoduced
            else:
                self.subjects = subjects

        self.data_path = data_path
        self.dataset = dataset
        self.split = split
        self.n_subjects = len(subjects)
        self.acc = acc
        self.filt = filt
        self.filt_path = filt_path
        self.fixed = fixed
        self.rnd_trials = rnd_trials
        self.norm_stim = norm_stim
        self.population = population
        self.hop = hop
       
        if dataset == 'fulsang':
            self.eeg, self.stima, self.stimb = self.get_Fulsang_data()
        elif dataset == 'jaulab':
            self.eeg, self.stima, self.stimb = self.get_Jaulab_data()
        elif dataset == 'skl':
            self.eeg, self.stima = self.get_SKL_data()
            self.stimb = None # No unattended stim on SparKuLee experiment
        else:
            raise ValueError('Introduce a valid dataset name between fulsang or skl')

        self.window = window
        self.n_samples = self.eeg.shape[1]

    def get_Fulsang_data(self):

        eeg = []
        stima = []
        stimb = []

        n_trials = 50 # 50 trials of 50s per subject in Fulsang dataset
        if not self.population:
            trials = get_trials(self.split, n_trials)
        else:
            trials = get_population_trials(self.split, n_trials)

        self.trials = trials

        for subject in self.subjects:
        
            preproc_data = scipy.io.loadmat(os.path.join(self.data_path ,subject + '_data_preproc.mat'))

            # Array con n trials y dentro las muestras de audio y eeg
            stima_data = preproc_data['data']['wavA'][0,0][0,trials]
            stimb_data = preproc_data['data']['wavB'][0,0][0,trials]
            n_trial = len(trials)

            if self.filt:
                eeg_data = np.load(os.path.join(self.filt_path, subject+'_data_filt.npy'), allow_pickle=True)[trials]
            else:
                eeg_data = preproc_data['data']['eeg'][0,0][0,trials]

            # si hay mas canales de la cuenta selecciono los 64 primeros
            if eeg_data[0].shape[1] > 64:
                eeg_data = [trial[:,:64] for trial in eeg_data]
        
            # Concatenar en un tensor todas os trials del sujeto (muestras * trials, canales) => (T * N, C).T => (C, T * N)
            eeg.append(torch.hstack([normalize_eeg(torch.tensor(eeg_data[trial]).T) for trial in range(n_trial)]))
            stima.append(torch.squeeze(torch.vstack([torch.tensor(stima_data[trial]) for trial in range(n_trial)])) if not self.norm_stim else torch.squeeze(normalize_stim(torch.vstack([torch.tensor(stima_data[trial]) for trial in range(n_trial)]))))
            stimb.append(torch.squeeze(torch.vstack([torch.tensor(stimb_data[trial]) for trial in range(n_trial)])) if not self.norm_stim else torch.squeeze(normalize_stim(torch.vstack([torch.tensor(stimb_data[trial]) for trial in range(n_trial)]))))

        # Concateno en un tensor global la información de los SUJETOS INDICADOS
        return torch.hstack(eeg), torch.cat(stima), torch.cat(stimb)

    def get_SKL_data(self):
    
        eeg = []
        stim = []
        gpu = True

        subj_idx = [get_SKL_subj_idx(subj) for subj in self.subjects]
        filelist = os.listdir(self.data_path)
        n_files = len(filelist)

        # Subject specific mode
        if not self.population:
            for subj in subj_idx:
                for n, file in enumerate(filelist):
                    chunks = file.split('_')
                    # Cargo la información del sujeto dependiendo del split
                    if self.split == chunks[0] and subj == chunks[2]:
                        data = torch.tensor(np.load(os.path.join(self.data_path, file)))
                        if 'eeg' in chunks[-1]:
                            eeg.append(data)
                        elif 'envelope' in chunks[-1]:
                            stim.append(data)

        # Subject independent / population mode
        else:
            for subj in subj_idx:
                # Carga por lotes para la operación torch.cat: ayuda al rendimiento en especial al entrenar
                print(f'Gathering data from subject {subj} on {self.split} loader')
                eeg_subj = []
                stim_subj = []
                for n, file in enumerate(filelist):
                    chunks = file.split('_')
                    n_subj_files = len([file for file in filelist if subj in file]) # get number of files of each subject
                    # Cargo la información del sujeto (todos los splits)
                    if subj == chunks[2]:
                        data = torch.tensor(np.load(os.path.join(self.data_path, file)))
                        if 'eeg' in chunks[-1]:
                            eeg_subj.append(data)
                        elif 'envelope' in chunks[-1]:
                            stim_subj.append(data)
                cat_eeg = torch.cat(eeg_subj, dim=0)
                cat_stim = torch.cat(stim_subj, dim=0)
                half_samples = cat_eeg.shape[0] // 2
                # Totalidad de muestras para el entrenamiento y carga por lotes
                if self.split =='train':
                    eeg.append(cat_eeg)
                    stim.append(cat_stim)
                    gc.collect()
                # La mitad de muestras para val y train del sujeto excluido en el entrenamiento
                elif self.split =='val':
                    eeg.append(cat_eeg[:half_samples, :])
                    stim.append(cat_stim[:half_samples, :])
                elif self.split =='test':
                    eeg.append(cat_eeg[half_samples:, :])
                    stim.append(cat_stim[half_samples:, :])
                else: raise ValueError('Introduce a valid split name between train val or test')
        
        eeg_cat = torch.cat(eeg).T
        stima_cat = torch.squeeze(torch.cat(stim))
        
        return eeg_cat, stima_cat
    
    def get_Jaulab_data(self):

        eeg = []
        stima = []
        stimb = []

        n_trials = 96 # 96 trials of 26s per subject in Fulsang dataset
        if not self.population:
            trials = get_trials(self.split, n_trials)
        else:
            trials = get_population_trials(self.split, n_trials)

        self.trials = trials

        for subject in self.subjects:
        
            preproc_data = scipy.io.loadmat(os.path.join(self.data_path ,subject + '_preproc.mat'))

            # Array con n trials y dentro las muestras de audio y eeg
            stima_data = preproc_data['data']['wavA'][0,0][0,trials]
            stimb_data = preproc_data['data']['wavB'][0,0][0,trials]
            n_trial = len(trials)

            if self.filt:
                eeg_data = np.load(os.path.join(self.filt_path, subject+'_data_filt.npy'), allow_pickle=True)[trials]
            else:
                eeg_data = preproc_data['data']['eeg'][0,0][0,trials]

            # Normalizar eeg
            norm_eeg = [normalize_eeg(torch.tensor(eeg_data[trial]).T) for trial in range(n_trial)]

            # Añadir canales con zero padding si estos no llegan a 64: puede haber sujetos con 63, 62 o 61 electrodos
            n_channels = eeg_data[0].shape[1] 
            rest_channels = 61 - n_channels
            zero_channels = torch.zeros((rest_channels, norm_eeg[0].shape[1]))
            zero_eeg = [torch.cat((norm_eeg[trial], zero_channels), dim=0) for trial in range(n_trial)]
            # zero_eeg = norm_eeg

            # Concatenar en un tensor todas os trials del sujeto (muestras * trials, canales) => (T * N, C).T => (C, T * N)
            eeg.append(torch.hstack([zero_eeg[trial] for trial in range(n_trial)]))
            stima.append(torch.squeeze(torch.vstack([torch.tensor(stima_data[trial]) for trial in range(n_trial)])) if not self.norm_stim else torch.squeeze(normalize_stim(torch.vstack([torch.tensor(stima_data[trial]) for trial in range(n_trial)]))))
            stimb.append(torch.squeeze(torch.vstack([torch.tensor(stimb_data[trial]) for trial in range(n_trial)])) if not self.norm_stim else torch.squeeze(normalize_stim(torch.vstack([torch.tensor(stimb_data[trial]) for trial in range(n_trial)]))))

        # Concateno en un tensor global la información de los SUJETOS INDICADOS
        return torch.hstack(eeg), torch.cat(stima), torch.cat(stimb)
    
    def __len__(self):
        return (self.n_samples - self.window) // self.hop
    

    def __getitem__(self, idx):

        start = idx * self.hop
        end = start + self.window

        eeg = self.eeg[:, start:end] 
        stima = self.stima[start] # only returning one sample
        
        if self.acc:
            stimb = self.stimb[start] # only returning one sample
            return {'eeg':eeg, 'stima':stima, 'stimb':stimb}
        else:
            return {'eeg':eeg, 'stima':stima}