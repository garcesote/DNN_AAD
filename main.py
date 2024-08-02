from pipeline.training_functions import train_dnn, train_ridge, leave_one_out_ridge
from pipeline.eval_functions import decode_attention, eval_dnn, eval_ridge

'''
    MAIN SCRIPT: Train and evaluate the different datasets on the different models fro the subject specif case. Select the subjects
    you want yoor network to train on
'''

models = ['CNN']
datasets = ['fulsang']

# Path where the data is located
paths = {'hugo_path' : "C:/Users/jaulab/Desktop/AAD/Data/Hugo_2022/hugo_preproc_data",
        'fulsang_path' : 'C:/Users/jaulab/Desktop/AAD/Data/Fulsang_2017/DATA_preproc',
        'jaulab_path' : 'C:/users/jaulab/Desktop/AAD/Data/Jaulab_2024/PreprocData_ICA',
        'fulsang_filt_path' : 'C:/Users/jaulab/Desktop/AAD/Data/Fulsang_2017/DATA_filtered',
        'jaulab_filt_path' : 'C:/users/jaulab/Desktop/AAD/Data/Jaulab_2024/DATA_filtered'
}

# Select the date for saving models and metrics
date = '01_08'

mdl_save_path = 'Results/models'
metrics_save_path = 'Results/train_metrics'
results_save_path = 'Results/eval_metrics/'
accuracies_save_path = 'Results/decode_accuracy'

subject = 'S13'

# # TRAIN DNN
for model in models:
    for dataset in datasets:
        data_path = paths[dataset+'_path']
        train_dnn(model=model, dataset=dataset, subjects=subject, data_path=data_path, key=date, mdl_save_path=mdl_save_path,
                  metrics_save_path=metrics_save_path, max_epoch=10, filt_path=paths[dataset+'_filt_path'])

# # EVALUATE DNN
# for model in models:
#     for dataset in datasets:
#         data_path = paths[dataset+'_path']
#         eval_dnn(model, dataset, subject, data_path, results_save_path, mdl_save_path, date , filt_path=paths[dataset+'_filt_path'])

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

# leave_one_out_ridge(dataset='fulsang', datapath=paths['fulsang_path'], window = 50, original=True, subject='S12', save_path='x')