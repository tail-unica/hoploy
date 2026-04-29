import argparse
import glob
import os
import shutil

from transformers.trainer_utils import get_last_checkpoint
from hopwise.quick_start import run_hopwise

# docker compose run train -c autism_test_hoploy -d autism
# CHECKPOINT_CONFIG=autism_test_hoploy DATASET=autism docker compose up train
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default=os.environ.get("CHECKPOINT_CONFIG", "autism_test_hoploy"))
parser.add_argument("-d", "--dataset", default=os.environ.get("DATASET", "autism"))
parser.add_argument("-m", "--model", default=os.environ.get("MODEL", "PEARLM")) # PEARLM, KGGLM
args = parser.parse_args()

config_name, dataset, model = args.config, args.dataset, args.model

checkpoint_dir = f"/app/checkpoints/{config_name}"
config_file    = f"{checkpoint_dir}/config.yaml"
data_path      = "/app/datasets"

# kgglm pretrain + finetune
def run_kgglm():
    # Phase 1: pretrain
    run_hopwise(
        model=model,
        dataset=dataset,
        config_file_list=[config_file],
        saved=True,
        config_dict={
            "train_stage": "pretrain",
            "data_path": data_path,
            "checkpoint_dir": checkpoint_dir,
        },
    )

    # Trova il checkpoint HF pretrained salvato da hopwise:
    # <checkpoint_dir>/huggingface-*-pretrained-<N>.pth/checkpoint-<step>/
    # Usa il pattern con trattino prima del numero per escludere pretrained.pth (dataloaders)
    # e ordina per mtime per prendere il più recente.
    candidates = sorted(
        glob.glob(f"{checkpoint_dir}/huggingface-*-pretrained-*.pth"),
        key=os.path.getmtime,
    )
    if not candidates:
        raise RuntimeError(f"No pretrained checkpoint found in {checkpoint_dir}")
    pre_model_path = get_last_checkpoint(candidates[-1])
    if pre_model_path is None:
        raise RuntimeError(f"No valid checkpoint-* subdir found in {candidates[-1]}")

    # Phase 2: finetune
    run_hopwise(
        model=model,
        dataset=dataset,
        config_file_list=[config_file],
        saved=True,
        config_dict={
            "train_stage": "finetune",
            "pre_model_path": pre_model_path,
            "data_path": data_path,
            "checkpoint_dir": checkpoint_dir,
        },
    )

    # Elimina tutte le directory pretrained intermedie dopo il finetune
    for d in glob.glob(f"{checkpoint_dir}/huggingface-*-pretrained-*.pth"):
        shutil.rmtree(d, ignore_errors=True)


def run_default():
    run_hopwise(
        model=model,
        dataset=dataset,
        config_file_list=[config_file],
        saved=True,
        config_dict={
            "data_path": data_path,
            "checkpoint_dir": checkpoint_dir,
        },
    )

map = {
    "KGGLM": run_kgglm,
}

if model in map:
    map[model]()
else:
    run_default()
