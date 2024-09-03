from pipeline.training_functions import train_dnn, train_ridge, leave_one_out_ridge
from pipeline.eval_functions import decode_attention, eval_dnn, eval_ridge

'''
    MAIN SCRIPT: Train and evaluate the different datasets on the different models fro the subject specif case. Select the subjects
    you want yoor network to train on
'''

# global_path = 'd:/AAD_EEG/Data' 
global_path = 'C:/Users/jaulab/Desktop/AAD/Data'

# Path where the data is located
paths = {'hugo_path' : global_path + "/Hugo_2022/hugo_preproc_data",
        'fulsang_path' : global_path + '/Fulsang_2017/DATA_preproc',
        'jaulab_path' : global_path + '/Jaulab_2024/PreprocData_ICA',
        'fulsang_filt_path' : global_path + '/Fulsang_2017/DATA_filtered',
        'jaulab_filt_path' : global_path + '/Jaulab_2024/DATA_filtered'
}

window = 128 # 2s

# Path where the data is located
ds_subjects = {'hugo_subj' : ['S'+str(n) for n in range(1, 13)],
        'fulsang_subj' : ['S'+str(n) for n in range(1, 19)],
        'jaulab_subj' : list(set(['S'+str(n) for n in range(1, 18)]))
}

models = ['FCNN', 'CNN']
datasets = ['hugo','fulsang','jaulab']

# Select the date for saving models and metrics
key = '09_08_win_'+str(window//64)+'s'

mdl_save_path = 'ResultsWindows/'+key+'/models'
metrics_save_path = 'ResultsWindows/'+key+'/train_metrics'
results_save_path = 'ResultsWindows/'+key+'/eval_metrics'
accuracies_save_path = 'ResultsWindows/'+key+'/decode_accuracy'

# TRAIN DNN
for model in models:
    for dataset in datasets:
        data_path = paths[dataset+'_path']
        window = window * 2 if dataset == 'hugo' else window
        filt_path = None if dataset=='hugo' else paths[dataset+'_filt_path']
        train_dnn(model=model, dataset=dataset, subjects=ds_subjects[dataset+'_subj'], window_len=window, data_path=data_path, mdl_save_path=mdl_save_path,
                    metrics_save_path=metrics_save_path, max_epoch=200, filt=True, filt_path=filt_path)
# EVALUATE DNN
for model in models:
    for dataset in datasets:
        data_path = paths[dataset+'_path']
        window = window * 2 if dataset == 'hugo' else window
        filt_path = None if dataset=='hugo' else paths[dataset+'_filt_path']
        eval_dnn(model, dataset, ds_subjects[dataset+'_subj'], window, data_path, results_save_path, mdl_save_path, 
                filt=True, filt_path=filt_path)

# LINEAR MODEL: RIDGE
for dataset in datasets:
    data_path = paths[dataset+'_path']
    window = window * 2 if dataset == 'hugo' else window
    filt_path = None if dataset=='hugo' else paths[dataset+'_filt_path']
    train_ridge(dataset, ds_subjects[dataset+'_subj'], data_path, mdl_save_path=mdl_save_path, original=True, 
                filt=True, filt_path=filt_path)
    eval_ridge(dataset, ds_subjects[dataset+'_subj'], window, data_path, mdl_save_path, dst_save_path= results_save_path, original=True, 
            filt=True, filt_path=filt_path)

# DECODING ACCURACY
window_lenghts = [128, 640, 1600, 3200]
models = ['CNN','FCNN','Ridge','Ridge_Original']
datasets = ['jaulab', 'fulsang']
for dataset in datasets:
    for model in models:
        for win in window_lenghts:
            decode_attention(model, dataset, ds_subjects[dataset+'_subj'], win, paths[dataset+'_path'], mdl_save_path, 
                            accuracies_save_path, filt_path=paths[dataset+'_filt_path'])

# leave_one_out_ridge(dataset='fulsang', datapath=paths['fulsang_path'], window = 50, original=True, subject='S12', save_path='x')