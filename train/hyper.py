import argparse
import glob
import os
import shutil

from hopwise.trainer import HyperTuning
from hopwise.quick_start import objective_function

# docker compose run train -c autism_test_hoploy -d autism
# CHECKPOINT_CONFIG=autism_test_hoploy DATASET=autism docker compose up train
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default=os.environ.get("CHECKPOINT_CONFIG", "movielens"))
parser.add_argument("-d", "--dataset", default=os.environ.get("DATASET", "movielens"))
parser.add_argument("-m", "--model", default=os.environ.get("MODEL", "PEARLM")) # PEARLM, KGGLM
args = parser.parse_args()

config_name, dataset, model = args.config, args.dataset, args.model

checkpoint_dir = f"/app/checkpoints/{config_name}"
config_file    = f"{checkpoint_dir}/config.yaml"
params_file    = f"{checkpoint_dir}/model.hyper"
data_path      = "/app/datasets"

hp = HyperTuning(
    objective_function=objective_function,
    algo='exhaustive',
    early_stop=10,
    max_evals=100,
    params_file=params_file,
    fixed_config_file_list=[config_file],
)

# run
hp.run()

# export result to the file
hp.export_result(output_file=f'{checkpoint_dir}/hyper.result')

# print best parameters
print('best params: ', hp.best_params)
# print best result
print('best result: ')
print(hp.params2result[hp.params2str(hp.best_params)])
