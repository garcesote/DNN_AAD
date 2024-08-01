from scipy.signal import firwin, lfilter, filtfilt
import numpy as np
import scipy
import h5py
import os

def lp_hamming_fir(signal, fs = 64, cutoff_freq = 8, num_taps=1651):

    # Filter design LP FIR type 1 hamming window
    coef = firwin(num_taps, cutoff_freq / (0.5 * fs), window='hamming', pass_zero='lowpass')
    filtered_signal = filtfilt(coef, [1.0], signal)
    return filtered_signal

def hp_hamming_fir(signal, fs = 64, cutoff_freq = 0.5, num_taps=825):
    
    # Filter design HP FIR type 1 hamming window
    coef = firwin(num_taps, cutoff_freq / (0.5 * fs), window='hamming', pass_zero='highpass')
    filtered_signal = filtfilt(coef, [1.0], signal)
    return filtered_signal


paths = {'hugo_path' : "C:/Users/jaulab/Desktop/AAD/Data/Hugo_2022/hugo_preproc_data",
        'fulsang_path' : 'C:/Users/jaulab/Desktop/AAD/Data/Fulsang_2017/DATA_preproc',
        'jaulab_path' : 'C:/users/jaulab/Desktop/AAD/Data/Jaulab_2024/PreprocData_ICA'
}

# FILTER JAULAB

subjects = ['S'+str(n+1) for n in range(17)]

save_path = "C:/Users/jaulab/Desktop/AAD/Data/Jaulab_2024/DATA_filtered"
if not os.path.exists(save_path):
    os.makedirs(save_path)

for subj in subjects:

    print(f'Filtering subject {subj} on jaulab dataset...')
    data_path = os.path.join(paths['jaulab_path'] ,subj + '_preproc')
    preproc_data = scipy.io.loadmat(data_path)

    # Array con todos los trials y dentro las muestras de audio y eeg
    eeg_data = preproc_data['data']['eeg'][0,0][0,:]
    trials = len(eeg_data)
    eeg_filtered = np.zeros_like(eeg_data)
    samples = eeg_data[0].shape[0]
    eeg_dataset = np.hstack([eeg_signal.T for eeg_signal in eeg_data])
    eeg_dataset = lp_hamming_fir(eeg_dataset)
    eeg_dataset = hp_hamming_fir(eeg_dataset)
    for trial in range(trials):
        eeg_filtered[trial] = eeg_dataset[:, samples*trial:samples*(trial+1)].T

    np.save(os.path.join(save_path, subj+'_data_filt.npy'), eeg_filtered)

# FILTER FULSANG

subjects = ['S'+str(n+1) for n in range(18)]

save_path = "C:/Users/jaulab/Desktop/AAD/Data/Fulsang_2017/DATA_filtered"
if not os.path.exists(save_path):
    os.makedirs(save_path)

for subj in subjects:

    print(f'Filtering subject {subj} on fulsang dataset...')
    data_path = os.path.join(paths['fulsang_path'] ,subj + '_data_preproc.mat')
    preproc_data = scipy.io.loadmat(data_path)

    # Array con todos los trials y dentro las muestras de audio y eeg
    eeg_data = preproc_data['data']['eeg'][0,0][0,:]
    trials = len(eeg_data)
    eeg_filtered = np.zeros_like(eeg_data)
    samples = eeg_data[0].shape[0]
    eeg_dataset = np.hstack([eeg_signal.T for eeg_signal in eeg_data])
    eeg_dataset = lp_hamming_fir(eeg_dataset)
    eeg_dataset = hp_hamming_fir(eeg_dataset)
    for trial in range(trials):
        eeg_filtered[trial] = eeg_dataset[:, samples*trial:samples*(trial+1)].T

    np.save(os.path.join(save_path, subj+'_data_filt.npy'), eeg_filtered)





    




