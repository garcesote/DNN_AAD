from pipeline.training_functions import train_dnn, train_ridge


# REPRODUCING TRAINING DNN RESULTS
models = ['FCNN', 'CNN']
datasets = ['hugo','Fulsang']

# Path where the data is located
hugo_path = "C:/Users/jaulab/Desktop/AAD/DNN_AAD/hugo_preproc_data"
fulsang_path = 'C:/Users/jaulab/Desktop/AAD/Fulsang_2017/DATA_preproc'

# Select the date for saving models and metrics
date = '16_07'

mdl_save_path = 'Results/models'
metrics_save_path = 'Results/train_metrics'


for model in models:
    for dataset in datasets:
        data_path = hugo_path if dataset == 'hugo' else fulsang_path
        train_dnn(model=model, dataset=dataset, data_path=data_path, date=date, mdl_save_path=mdl_save_path, metrics_save_path=metrics_save_path)

# TRAINIG RIDGE
for dataset in datasets:
    train_ridge(dataset, data_path, mdl_save_path=mdl_save_path, date=date, original=False)