import subprocess

# Definir los parámetros que quieres pasarle a train.py
params_list = [
    {'model': 'CNN', 'batch_size': 128, 'block_size':64, 'dataset': 'jaulab', 'filt': True, 'fixed':False, 'rnd_trials': False},
    {'model': 'FCNN', 'batch_size': 128, 'block_size':64, 'dataset': 'fulsang', 'filt': True, 'fixed':False, 'rnd_trials': False},
    {'model': 'FCNN', 'batch_size': 128, 'block_size':64, 'dataset': 'jaulab', 'filt': True, 'fixed':False, 'rnd_trials': False},
    {'model': 'CNN', 'batch_size': 128, 'block_size':64, 'dataset': 'fulsang', 'filt': True, 'fixed':False, 'rnd_trials': False},
]

for params in params_list:
    # Crear el comando para ejecutar train.py con los parámetros
    cmd = [
        "py", "train_models.py", 
        "--model", str(params['model']),
        "--batch_size", str(params['batch_size']), 
        "--block_size", str(params['block_size']), 
        "--dataset", str(params['dataset']),
        "--filt", str(params['filt']),
        "--fixed", str(params['fixed']),
        "--rnd_trials", str(params['rnd_trials']),
    ]

    print(cmd)
    # Llamar a train.py
    subprocess.run(cmd)