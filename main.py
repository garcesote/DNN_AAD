from pipeline.training_functions import train_dnn, train_ridge, leave_one_out_ridge
from pipeline.eval_functions import decode_attention, eval_dnn, eval_ridge

# REPRODUCING TRAINING DNN RESULTS
models = ['FCNN', 'CNN']
datasets = ['fulsang','hugo']

# Path where the data is located
paths = {'hugo_path' : "C:/Users/jaulab/Desktop/AAD/Data/Hugo_2022/hugo_preproc_data",
        'fulsang_path' : 'C:/Users/jaulab/Desktop/AAD/Data/Fulsang_2017/DATA_preproc',
        'jaulab_path' : 'C:/users/jaulab/Desktop/AAD/Data/Jaulab_2024/PreprocData_ICA'
}

# Select the date for saving models and metrics
date = '19_07'

mdl_save_path = 'Results/models'
metrics_save_path = 'Results/train_metrics'
results_save_path = 'Results/eval_metrics/'
accuracies_save_path = 'Results/decode_accuracy'


# TRAIN DNN
for model in models:
    for dataset in datasets:
        data_path = paths[dataset+'_path']
        train_dnn(model=model, dataset=dataset, data_path=data_path, key=date, mdl_save_path=mdl_save_path, metrics_save_path=metrics_save_path, max_epoch=200)

# EVALUATE DNN
for model in models:
    for dataset in datasets:
        data_path = paths[dataset+'_path']
        eval_dnn(model, dataset, data_path, results_save_path, mdl_save_path, date)

# # TRAINIG RIDGE
for dataset in datasets:
    data_path = paths[dataset+'_path']
    train_ridge(dataset, data_path, mdl_save_path=mdl_save_path, key=date, original=True)
    eval_ridge(dataset, data_path, mdl_save_path, key= date, dst_save_path= results_save_path, original=True)
    train_ridge(dataset, data_path, mdl_save_path=mdl_save_path, key=date, original=False)
    eval_ridge(dataset, data_path, mdl_save_path, key= date, dst_save_path= results_save_path, original=False) 

# # DECODING ACCURACY
window_lenghts = [128, 640, 1600]
models = ['CNN','FCNN','Ridge','Ridge_Original']
dataset = 'fulsang'
for model in models:
    for win in window_lenghts:
        decode_attention(model, dataset, win, paths[dataset+'_path'], mdl_save_path, accuracies_save_path, key=date)

# leave_one_out_ridge(dataset='fulsang', datapath=paths['fulsang_path'], window = 50, original=True, subject='S12', save_path='x')