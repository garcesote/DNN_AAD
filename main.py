from pipeline.training_functions import train_dnn, train_ridge, leave_one_out_ridge
from pipeline.eval_functions import decode_attention, eval_dnn, eval_ridge

'''
    MAIN SCRIPT: Train and evaluate the different datasets on the different models fro the subject specif case. Select the subjects
    you want yoor network to train on
'''

global_path = 'd:/AAD_EEG/Data' 
# global_path = 'C:/Users/jaulab/Desktop/AAD/Data'

# Path where the data is located
paths = {'hugo_path' : global_path + "/Hugo_2022/hugo_preproc_data",
        'fulsang_path' : global_path + '/Fulsang_2017/DATA_preproc',
        'jaulab_path' : global_path + '/Jaulab_2024/PreprocData_ICA',
        'fulsang_filt_path' : global_path + '/Fulsang_2017/DATA_filtered',
        'jaulab_filt_path' : global_path + '/Jaulab_2024/DATA_filtered'
}

windows = [128, 640, 1600, 3200] # 2s, 10s, 25s, 50s

for window in windows:

    models = ['FCNN', 'CNN']
    datasets = ['hugo', 'fulsang', 'jaulab']

    # Select the date for saving models and metrics
    date = '09_08_win_'+str(window//64)+'s'

    mdl_save_path = 'Results/models'
    metrics_save_path = 'Results/train_metrics'
    results_save_path = 'Results/eval_metrics'
    accuracies_save_path = 'Results/decode_accuracy'

    # Path where the data is located
    ds_subjects = {'hugo_subj' : ['S'+str(n) for n in range(1, 13)],
            'fulsang_subj' : ['S'+str(n) for n in range(1, 19)],
            'jaulab_subj' : list(set(['S'+str(n) for n in range(1, 18)]))
    }

    # TRAIN DNN
    for model in models:
        for dataset in datasets:
            data_path = paths[dataset+'_path']
            window = window * 2 if dataset == 'hugo' else window
            train_dnn(model=model, dataset=dataset, subjects=ds_subjects[dataset+'_subj'], window_len=window, data_path=data_path, key=date, mdl_save_path=mdl_save_path,
                    metrics_save_path=metrics_save_path, max_epoch=200, filt=True, filt_path=paths[dataset+'_filt_path'])

    # EVALUATE DNN
    for model in models:
        for dataset in datasets:
            data_path = paths[dataset+'_path']
            window = window * 2 if dataset == 'hugo' else window
            eval_dnn(model, dataset, window, ds_subjects[dataset+'_subj'], data_path, results_save_path, mdl_save_path, date, 
                    filt=True, filt_path=paths[dataset+'_filt_path'])

    # LINEAR MODEL: RIDGE
    for dataset in datasets:
        data_path = paths[dataset+'_path']
        window = window * 2 if dataset == 'hugo' else window
        train_ridge(dataset, ds_subjects[dataset+'_subj'], data_path, mdl_save_path=mdl_save_path, key=date, original=True, 
                    filt=True, filt_path=paths[dataset+'_filt_path'])
        eval_ridge(dataset, ds_subjects[dataset+'_subj'], window, data_path, mdl_save_path, key= date, dst_save_path= results_save_path, original=True, 
                filt=True, filt_path=paths[dataset+'_filt_path'])
        train_ridge(dataset, ds_subjects[dataset+'_subj'], window, data_path, mdl_save_path=mdl_save_path, key=date, original=False, 
                    filt=True, filt_path=paths[dataset+'_filt_path'])
        eval_ridge(dataset, ds_subjects[dataset+'_subj'], data_path, mdl_save_path, key= date, dst_save_path= results_save_path, original=False, 
                filt=True, filt_path=paths[dataset+'_filt_path']) 

    # DECODING ACCURACY
    window_lenghts = [128, 640, 1600, 3200]
    models = ['CNN','FCNN','Ridge','Ridge_Original']
    datasets = ['jaulab', 'fulsang']
    for dataset in datasets:
        for model in models:
            for win in window_lenghts:
                decode_attention(model, dataset, ds_subjects[dataset+'_subj'], win, paths[dataset+'_path'], mdl_save_path, 
                                accuracies_save_path, key=date, filt_path=paths[dataset+'_filt_path'])

# leave_one_out_ridge(dataset='fulsang', datapath=paths['fulsang_path'], window = 50, original=True, subject='S12', save_path='x')