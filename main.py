from pipeline.training_functions import train_dnn, train_ridge
from pipeline.eval_functions import decode_attention, eval_dnn, eval_ridge

# REPRODUCING TRAINING DNN RESULTS
models = ['FCNN', 'CNN']
datasets = ['jaulab','fulsang','hugo']

# Path where the data is located
hugo_path = "C:/Users/jaulab/Desktop/AAD/Data/Hugo_2022/hugo_preproc_data"
fulsang_path = 'C:/Users/jaulab/Desktop/AAD/Data/Fulsang_2017/DATA_preproc'
jaulab_path = 'C:/users/jaulab/Desktop/AAD/Data/Jaulab_2024/PreprocData_ICA'

# Select the date for saving models and metrics
date = '17_07'

mdl_save_path = 'Results/models'
metrics_save_path = 'Results/train_metrics'
results_save_path = 'Results/eval_metrics/'
accuracies_save_path = 'Results/decode_accuracy'


# TRAIN DNN
# for model in models:
#     for dataset in datasets:
#         data_path = hugo_path if dataset == 'hugo' else fulsang_path
#         train_dnn(model=model, dataset=dataset, data_path=data_path, date=date, mdl_save_path=mdl_save_path, metrics_save_path=metrics_save_path, max_epoch=1)


# TRAINIG RIDGE
dataset = 'jaulab'
data_path = jaulab_path
# train_ridge(dataset, data_path, mdl_save_path=mdl_save_path, key=date, original=True)
# eval_ridge(dataset, data_path, mdl_save_path, key= date, dst_save_path= results_save_path, original=True)

# results_save_path = 'Results/eval_metrics/'

# # EVALUATE FUNCTION
# for model in models:
#     for dataset in datasets:
#         data_path = hugo_path if dataset == 'hugo' else fulsang_path
#         eval_dnn(model, dataset, data_path, results_save_path, mdl_save_path, date)

# dataset = 'jaulab'
# model = 'FCNN'

# for dataset in datasets:
#     for model in models:
#         train_dnn(model, dataset, jaulab_path, metrics_save_path, date, mdl_save_path, max_epoch=200)
#         eval_dnn(model, dataset, jaulab_path, results_save_path, mdl_save_path, date)

# DECODING ACCURACY
# models = ['FCNN', 'CNN']
window_lenghts = [128, 640, 1600]
dataset = 'jaulab'
model = 'CNN'

# for model in models:
for win in window_lenghts:
    decode_attention(model, dataset, win, jaulab_path, mdl_save_path, accuracies_save_path, key=date)