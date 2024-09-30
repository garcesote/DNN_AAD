import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from utils.dnn import CNN, FCNN
import os
import scipy 
import json
from utils.custom_dataset import CustomDataset
from tqdm import tqdm
import argparse

def main(
        model: str,
        batch_size: int, 
        block_size: int,
        dataset: str,
        population: bool,
        filt: bool,
        fixed: bool,
        rnd_trials: bool,
    ):

    mdl_name = f'{model}_30_09'
    print(mdl_name)

    # Saving path parameters
    global_path = 'C:/Users/jaulab/Desktop/AAD/deepdecoding'
    # global_path = r'C:\Users\garce\Desktop\proyecto_2024\Repos_2024\transformer\DecAccNet'
    key = 'subj_specific' if not population else 'population'
    mdl_save_path = global_path + '/Results_population/models'
    metrics_save_path = global_path + '/Results_population/metrics'

    # Select subjects you want your network to be trained on
    subjects = {
        'fulsang_subj': ['S'+str(n) for n in range(1, 19)],
        'skl_subj': ['S'+str(n) for n in range(1, 85)],
        # 'jaulab_subj' : ['S'+str(n) for n in range(1, 18) if n not in jaulab_excl_subj]
        'jaulab_subj' : ['S'+str(n) for n in range(1, 18)]
    }

    # Data path parameters
    global_path = 'C:/Users/jaulab/Desktop/AAD/Data'
    # global_path = 'C:/Users/garcia.127407/Desktop/DNN_AAD/Data'
    # global_path = 'D:/AAD_EEG/Data'

    paths = {'hugo_path': global_path + "/Hugo_2022/hugo_preproc_data",
            'fulsang_path': global_path + '/Fulsang_2017/DATA_preproc',
            'jaulab_path': global_path + '/Jaulab_2024/PreprocData_ICA',
            'fulsang_filt_path': global_path + '/Fulsang_2017/DATA_filtered',
            'jaulab_filt_path': global_path + '/Jaulab_2024/DATA_filtered',
            'jaulab_fix_path': global_path + '/Jaulab_2024/fixed_trials.npy',
            'skl_path': global_path + '/SKL_2023/split_data',
            'skl_filt_path': None,
    }

    channels = {
        'skl': 64,
        'fulsang': 64,
        'jaulab': 61
    }

    #------------------------------------------------------------------------------------------

    """

    Training parameters
    ------------------

    window_len: int
        number of samples of eeg selected to predict the stim

    batch_size: int
        batch size selected for the dataloader corresponding to the number of samples
        predicted with the model

    shuffle: bool
        select if you want your dataloader to pick randomly time-based the windows

    max_epoch: int
        maximum number of epoch during training

    scheduler_patience: int
        number of epoch until decreasing the lr by a factor of 0.5

    early_stopping_patience: int
        number of epoch until stop training when loss is not decreasing

    lr: float
        learning rate applied when training by the optimizer

    dropout: float
        dropout applied on each block of the network

    dataset: string
        select a dataset between 'skl', 'fulsang' or 'jaulab' to train the network on

    population: bool
        select if the model would be trained on the leave-one-out subject paradigm (True) 
        or on the specified subject (False) (subject specific/ subject independent)

    filt: bool
        select wether you want to select the filtered eeg from fulsang or jaulab
    
    fixed: bool
        in the case the dataset is "jaulab" select only the trials in which the stimulus is 
        fixed during the experiment. 

    rnd_trials: bool
        select if you want your trials to be selected randomly or assing them in order.
        In subject-specific mode trials are shuffled while in population mode, trials for eval.
        the excluded subject (val and test) are selected one-by-one.  

    compile: bool
        select if you want yor model to be faster (requires pytorch 2.0 not Windows) 

    """

    window_len = block_size
    hop = block_size // 4 # 75% overlapping windows
    batch_size = batch_size
    shuffle = True
    max_epoch = 200
    scheduler_patience = 2
    early_stopping_patience = 4
    dataset = dataset
    population = population # Attention! This parameter change the whole sim. (read doc)
    filt = filt
    fixed = fixed
    rnd_trials = rnd_trials
    compile = False

    torch.manual_seed(1) # Only applied when rnd_trials==True for selecting the trials 
                        # for train, test and val

    # Add extensions to the model name depending on the params
    if filt:
        mdl_name = mdl_name + '_filt'
    if rnd_trials:
        mdl_name = mdl_name + '_rnd'

    #------------------------------------------------------------------------------------------

    # model file_name (:0e cientific natation)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_results = {key: None for key in subjects[dataset+'_subj']}

    # selected_subjects = subjects[dataset+'_subj'][8:] if not embedding else subjects[dataset+'_subj']
    selected_subjects = subjects[dataset+'_subj']

    torch.manual_seed(1) # Used for random shuffle on stima and stimb

    n_chan = channels[dataset]

    # Only selected the first subjects for population independent mode
    for subj in selected_subjects:

        # LOAD THE MODEL
        if model == 'FCNN':
            mdl = FCNN(n_hidden = 3, dropout_rate=0.45, n_chan=n_chan, n_samples=block_size)
            optimizer = torch.optim.NAdam(mdl.parameters(), lr=1e-6, weight_decay = 1e-4)
        else:
            mdl = CNN(F1=8, D=8, F2=64, dropout=0.2, input_channels=n_chan, input_samples=block_size)
            optimizer = torch.optim.NAdam(mdl.parameters(), lr=2e-5, weight_decay = 1e-8)
        
        mdl.to(device)

        # Compile the model if possible (not possible on Windows)
        if compile:
            print("Compiling the model... (takes a ~minute)")
            mdl = torch.compile(mdl) # requires PyTorch 2.0

        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience, verbose=True)

        # LOAD THE DATA
        train_set = CustomDataset(dataset, paths[dataset+'_path'], 'train', subj, window=window_len, hop=hop, filt=filt, filt_path=paths[dataset+'_filt_path'], 
                                  population=population, fixed=fixed, rnd_trials = rnd_trials)
        val_set = CustomDataset(dataset, paths[dataset+'_path'], 'val', subj, window=window_len, hop=hop, filt=filt, filt_path=paths[dataset+'_filt_path'], 
                                population=population, fixed=fixed, rnd_trials = rnd_trials)

        train_loader = DataLoader(train_set, batch_size, shuffle=shuffle, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size, shuffle=shuffle, pin_memory=True)

        print(f'Training TamporalEncoder transformer with {dataset} data on subject {subj}...')

        # Early stopping parameters
        best_accuracy=0
        best_epoch=0

        train_loss = []
        val_loss = []

        # Training loop
        for epoch in range(max_epoch):
            
            # Stop after n epoch without imporving the val loss
            if epoch > best_epoch + early_stopping_patience:
                break

            mdl.train()
            train_accuracies = []

            # Initialize tqdm progress bar
            train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch}/{max_epoch}', leave=False, mininterval=0.5)

            for batch, data in enumerate(train_loader_tqdm):
                
                eeg = data['eeg'].to(device, dtype=torch.float)
                stima = data['stima'].to(device, dtype=torch.float)

                # Forward the model and calculate the loss corresponding to the neg. Pearson coef
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    y_hat, loss = mdl(eeg, targets = stima)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Append neg. loss corresponding to the coef. Pearson
                train_accuracies.append(-loss)

                # Actualize the state of the train loss
                train_loader_tqdm.set_postfix({'train_loss': loss.item()})

            mdl.eval()
            val_accuracies = []

            # Validation
            with torch.no_grad():

                for batch, data in enumerate(val_loader):

                    eeg = data['eeg'].to(device, dtype=torch.float)
                    stima = data['stima'].to(device, dtype=torch.float)

                    preds, loss = mdl(eeg, targets=stima)

                    val_accuracies.append(-loss)

            mean_val_accuracy = torch.mean(torch.hstack(val_accuracies)).item()
            mean_train_accuracy = torch.mean(torch.hstack(train_accuracies)).item()

            scheduler.step(mean_val_accuracy)

            # Logging metrics
            print(f'Epoch: {epoch} | Train accuracy: {mean_train_accuracy:.4f} | Val accuracy: {mean_val_accuracy:.4f}')

            train_loss.append(mean_train_accuracy)
            val_loss.append(mean_val_accuracy)

            # Save best results
            if mean_val_accuracy > best_accuracy or epoch == 0:
                # best_train_loss = mean_train_accuracy
                best_accuracy = mean_val_accuracy
                best_epoch = epoch
                best_state_dict = mdl.state_dict()
        
        
        dataset_filename = dataset+'_fixed' if fixed and dataset == 'jaulab' else dataset

        # Save best final model
        mdl_folder = os.path.join(mdl_save_path, dataset_filename+'_data', mdl_name)
        if not os.path.exists(mdl_folder):
            os.makedirs(mdl_folder)
        torch.save(
            best_state_dict, 
            os.path.join(mdl_folder, subj+f'_epoch={epoch}_acc={best_accuracy:.4f}.ckpt')
        )

        # Save corresponding train and val metrics
        val_folder = os.path.join(metrics_save_path, dataset_filename+'_data', mdl_name, 'val')
        if not os.path.exists(val_folder):
            os.makedirs(val_folder)
        train_folder = os.path.join(metrics_save_path, dataset_filename+'_data', mdl_name, 'train')
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)
        json.dump(train_loss, open(os.path.join(train_folder, subj+f'_train_loss_epoch={epoch}_acc={best_accuracy:.4f}'),'w'))
        json.dump(val_loss, open(os.path.join(val_folder, subj+f'_val_loss_epoch={epoch}_acc={best_accuracy:.4f}'),'w'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script")
    n_threads = 20
    torch.set_num_threads(n_threads)
    
    # Definir los argumentos que quieres aceptar
    parser.add_argument("--model", type=str, default='CNN', help="Select model between FCNN and CNN")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--block_size", type=int, default=64, help="Batch size")
    parser.add_argument("--dataset", type=str, default='fulsang', help="Dataset")
    parser.add_argument("--population", type=str, default='True', help="Population or Subject-Specific")
    parser.add_argument("--filt", type=str, default='False', help="EEG filtered")
    parser.add_argument("--fixed", type=str, default='False', help="Static Jaulab trials")
    parser.add_argument("--rnd_trials", type=str, default='False', help="Random trial selection")
    
    # Parsear los argumentos
    args = parser.parse_args()
    print(args)
    filt = False if args.filt == 'False' else True
    fixed = False if args.fixed == 'False' else True
    rnd_trials = False if args.rnd_trials == 'False' else True
    population = False if args.population == 'False' else True
    
    # Llamar a la función de entrenamiento con los argumentos
    main(args.model,
         args.batch_size,  
         args.block_size,
         args.dataset,
         population,
         filt,
         fixed,
         rnd_trials)