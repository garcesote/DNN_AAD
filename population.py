from pipeline.training_functions import train_dnn, train_ridge, leave_one_out_ridge
from pipeline.eval_functions import decode_attention, eval_dnn, eval_ridge

'''
    POPULATION SCRIPT: Train and validate models for the validation strategy of leaving-one-subject-out, where all subjcts except one are used
    for training and the excluded is used for validating and test the network.
'''
models = ['CNN', 'FCNN']
datasets = ['fulsang', 'jaulab']

# Path where the data is located
paths = {'hugo_path' : "C:/Users/jaulab/Desktop/AAD/Data/Hugo_2022/hugo_preproc_data",
        'fulsang_path' : 'C:/Users/jaulab/Desktop/AAD/Data/Fulsang_2017/DATA_preproc',
        'jaulab_path' : 'C:/users/jaulab/Desktop/AAD/Data/Jaulab_2024/PreprocData_ICA',
        'fulsang_filt_path' : 'C:/Users/jaulab/Desktop/AAD/Data/Fulsang_2017/DATA_filtered',
        'jaulab_filt_path' : 'C:/users/jaulab/Desktop/AAD/Data/Jaulab_2024/DATA_filtered'
}

# Select the date for saving models and metrics
date = '02_08'

mdl_save_path = 'Results_population/models'
metrics_save_path = 'Results_population/train_metrics'
results_save_path = 'Results_population/eval_metrics/'
accuracies_save_path = 'Results_population/decode_accuracy'

excluded_subj = set(['S1'])
# Excluding jaulab subjects with 60 electrodes instead of 61
excluded_jaulab_subj = set(['S13', 'S16'])

# Path where the data is located
subjects = {'hugo_subj' : list(set(['S'+str(n) for n in range(1, 14)]) - excluded_subj),
        'fulsang_subj' : list(set(['S'+str(n) for n in range(1, 19)]) - excluded_subj),
        'jaulab_subj' : list(set(['S'+str(n) for n in range(1, 18)]) - excluded_subj - excluded_jaulab_subj)
}

# TRAIN DNN
# for dataset in datasets:
#     for model in models:
#         data_path = paths[dataset+'_path']
#         train_dnn(model=model, dataset=dataset, subjects=subjects[dataset+'_subj'], data_path=data_path, key=date, mdl_save_path=mdl_save_path,
#                   metrics_save_path=metrics_save_path, max_epoch=200, population = True, filt_path=paths[dataset+'_filt_path'])

# EVALUATE DNN
for model in models:
    for dataset in datasets:
        data_path = paths[dataset+'_path']
        eval_dnn(model, dataset, list(excluded_subj), data_path, results_save_path, mdl_save_path, date , population=True, filt_path=paths[dataset+'_filt_path'])

# # TRAINIG RIDGE
# for dataset in datasets:
#     data_path = paths[dataset+'_path']
#     train_ridge(dataset, subject, data_path, mdl_save_path=mdl_save_path, key=date, original=True, filt_path=paths[dataset+'_filt_path'])
#     eval_ridge(dataset, subject, data_path, mdl_save_path, key= date, dst_save_path= results_save_path, original=True, filt_path=paths[dataset+'_filt_path'])
#     train_ridge(dataset, subject, data_path, mdl_save_path=mdl_save_path, key=date, original=False, filt_path=paths[dataset+'_filt_path'])
#     eval_ridge(dataset, subject, data_path, mdl_save_path, key= date, dst_save_path= results_save_path, original=False, filt_path=paths[dataset+'_filt_path']) 

# DECODING ACCURACY
# window_lenghts = [128, 640, 1600]
# window_lenghts = [3200]
# models = ['CNN','FCNN','Ridge','Ridge_Original']
# datasets = ['jaulab', 'fulsang']
# for dataset in datasets:
#     for model in models:
#         for win in window_lenghts:
#             decode_attention(model, dataset, subject, win, paths[dataset+'_path'], mdl_save_path, accuracies_save_path, key=date, filt_path=paths[dataset+'_filt_path'])