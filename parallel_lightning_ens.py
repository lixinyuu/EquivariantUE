import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

import argparse
import csv
import os
import time
from copy import deepcopy
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from torch.multiprocessing import Pool, set_start_method
from torch.utils.data import Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from uncertainty_ai.dataset import MD17, QM7X, FloatToTensorTransform, QM9P
from uncertainty_ai.metrics.calibration_scores import calibration_regression
from uncertainty_ai.metrics.force_calibration import (energy_recalibration,
                                                        calibration_regression_vector,
                                                        miscalibration_area_vector,
                                                        get_force_cov)
from uncertainty_ai.metrics.uct_utils import write_all_metrics
from training_lightning import LightningTraining, LightningTrainingDropout, LightningTrainingVariationalDropout
from uncertainty_ai.utils import get_model_args, has_tensor_cores
from ocpmodels.datasets import LmdbDataset

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def set_python_path():
    sys.path.insert(0, CURRENT_DIR)

def set_seeds(seed: int):
    """Set seed for reproducibility"""
    import os
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def save_args(args, path, name):
    config = deepcopy(args)
    with open(os.path.join(path, name + '.yml'), 'w') as f:
        hps = {key: val for key, val in config.items() if not isinstance(val, type)}
        yaml.dump(hps, f, default_flow_style=False)

def load_dataset(train_args):
    """Load dataset and create train/val/test splits with fixed seed for consistency"""
    # Set a fixed seed for dataset splitting to ensure all ensemble members use same splits
    set_seeds(42)  # Fixed seed for data splitting
    
    # load dataset
    dataset = QM7X("./dataset/QM7X", target= "dp", minima="global")
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    print("Train dataset size: ", len(train_dataset))
    print(train_dataset[0])
        
    return train_dataset, valid_dataset, test_dataset, dataset

def train_single_ensemble_member(ensemble_id, model_args, train_args, train_dataset, valid_dataset, test_dataset, dataset, base_path):
    """Train a single ensemble member"""
    print(f"Training ensemble member {ensemble_id}")
    
    # Create path for this ensemble member
    member_path = Path(base_path, f"{ensemble_id}")
    member_path.mkdir(parents=True, exist_ok=True)
    
    # Set unique seed for this ensemble member
    member_seed = train_args['base_seed'] + ensemble_id
    set_seeds(member_seed)
    set_python_path()
    
    # Set tensor cores
    if has_tensor_cores():
        torch.set_float32_matmul_precision('high')
        print("Using tensor cores")
    
    # Setup loggers
    writer = TensorBoardLogger(save_dir=member_path)
    csv_logger = CSVLogger(save_dir=member_path, name='csv')
    
    # Calculate energy statistics
    if len(train_dataset) < 5000:
        energy_mean = np.mean([d.y.numpy() for d in train_dataset])
        energy_std = np.std([d.y.numpy() for d in train_dataset])
    else:
        sample_idx = np.random.choice(len(train_dataset), 5000, replace=False)
        energy_mean = np.mean([train_dataset[i].y.numpy() for i in sample_idx])
        energy_std = np.std([train_dataset[i].y.numpy() for i in sample_idx])
    
    print(f"Ensemble member {ensemble_id}: Energy mean = {energy_mean:.4f}, Energy std = {energy_std:.4f}")
    
    train_args['normalizer_args']['mean'] = 0.0
    train_args['normalizer_args']['std'] = 1.0
    # Handle MACE model specific setup
    
    
    # Create model
    temp_model_args = deepcopy(model_args)
    del temp_model_args['name']
    gnn_model, final_model_args = get_model_args(model_args['name'], temp_model_args)
    
    # Save configuration
    save_args(final_model_args, member_path, 'model_args')
    save_args(train_args, member_path, 'train_args')
    
    device = 'cuda'
    
    # Initialize appropriate model type
    model = LightningTraining(gnn_model, final_model_args, train_args, writer, device, amp=train_args['amp'], init_model=True)

    
    # Setup trainer
    trainer = pl.Trainer(
        accelerator="cuda", 
        max_epochs=train_args['n_epoch'], 
        devices=1,  # Use single GPU for each ensemble member
        inference_mode=False, 
        precision="bf16-mixed" if train_args['amp'] else "32",
        logger=[writer, csv_logger]
    )
    
    # Train model
    trainer.fit(
        model, 
        DataLoader(train_dataset, **train_args['loader_tr_args']), 
        DataLoader(valid_dataset, **train_args['loader_te_args'])
    )
    
    # Make predictions on test set
    test_loader = DataLoader(test_dataset, **train_args['loader_te_args'])
    predictions = trainer.predict(model, test_loader)
    
    # Process predictions based on model type
    dp_pred = []
    dp_true = []
    dp_stds = []
    
    for idx, pred in enumerate(predictions):
        dp_pred_, dp_true_ = pred
        dp_pred.append(dp_pred_.detach().to("cpu"))
        dp_true.append(dp_true_.detach().to("cpu"))
    
    dp_stds = [torch.eye(3)[None, :, :].repeat(dp_pred_.size(0), 1, 1) for dp_pred_ in dp_pred]


    # Concatenate results
    dp_pred = torch.cat(dp_pred, dim=0)
    dp_true = torch.cat(dp_true, dim=0)
    
    # Save predictions for this ensemble member
    torch.save(dp_pred, member_path / "dipole_pred.pt")
    torch.save(dp_true, member_path / "dipole_true.pt")

    if len(dp_pred.shape) == 3: # it is sampling output
        dp_pred = dp_pred.mean(dim=1)
    dp_mae = float(torch.nn.functional.l1_loss(dp_pred, dp_true))
    print("dipole moment = {:.4f}".format(dp_mae))
    return member_path

@rank_zero_only
def ensemble_uncertainty_estimation(ensemble_paths, base_path):
    """Combine predictions from all ensemble members and calculate ensemble uncertainty"""
    print("Performing ensemble uncertainty estimation...")
    
    n_members = len(ensemble_paths)
    all_dp_pred = []
    dp_true = None
    
    # Load predictions from all ensemble members
    for i, member_path in enumerate(ensemble_paths):
        dp_pred = torch.load(member_path / "dipole_pred.pt")
        
        if i == 0:
            dp_true = torch.load(member_path / "dipole_true.pt")
        
        all_dp_pred.append(dp_pred)
    
    # Stack predictions: [n_members, n_samples, ...]
    all_dp_pred = torch.stack(all_dp_pred, dim=0)  # [n_members, n_samples, n_atoms*3]
    
    # Calculate ensemble mean predictions
    dp_pred_mean = torch.mean(all_dp_pred, dim=0)
    
    # Calculate ensemble uncertainties (standard deviation across ensemble members)
    # f_std_ensemble in the shape of [n_samples, n_atoms, 3]
    # change it to [n_atoms, n_samples, 3] for consistency
    dp_preds = all_dp_pred.permute(1, 0, 2)
    dp_stds = get_force_cov(dp_preds)
    dp_pred = dp_preds.mean(dim=1)  # [n_samples, n_atoms*3]
    
    
    f_mae = float(torch.nn.functional.l1_loss(dp_pred.norm(dim=-1), dp_true.norm(dim=-1)))
    f_l2dist = float((dp_pred - dp_true).norm(dim=-1).mean().item())
    f_miscalibration_area = miscalibration_area_vector(dp_pred, dp_stds, dp_true)
    f_calibration_score = calibration_regression_vector(dp_pred, dp_stds, dp_true)
    
    print("dp mae = {:.4f}".format(f_mae))
    print("dp miscalibration_area = {:.4f}".format(f_miscalibration_area))
    print("dp calibration score = {:.4f}".format(f_calibration_score))
    with open(Path(base_path, "ensemble_results.txt") ,'w') as fp:
        fp.write("dp mae = {:.4f}\n".format(f_mae))
        fp.write("dp miscalibration_area = {:.4f}\n".format(f_miscalibration_area))
        fp.write("dp calibration score = {:.4f}\n".format(f_calibration_score))
    dir_path = Path(base_path).parent
    seed = Path(base_path).name
    with open(Path(dir_path, "results.csv"), 'a') as f:
        f.write(f"{seed}, {f_mae}, {f_l2dist}, {f_miscalibration_area}, {f_calibration_score}\n")


    return {
        'dp_mae': f_mae,
        'dp_l2dist': f_l2dist,
        'dp_miscalibration_area': f_miscalibration_area,
        'dp_calibration_score': f_calibration_score,
    }

def ensemble_training(model_args, train_args, ppath, n_ensemble=5):
    """Main ensemble training function"""
    print(f"Starting ensemble training with {n_ensemble} members")
    
    # Create base path for ensemble
    
    path = Path(ppath, str(train_args['seed']))
    path.mkdir(parents=True, exist_ok=True)
    
    # Save ensemble configuration
    ensemble_config = {
        'n_ensemble': n_ensemble,
        'base_seed': train_args['seed'],
        'model_args': model_args,
        'train_args': train_args
    }
    save_args(ensemble_config, path, 'ensemble_config')
    
    # Load dataset with fixed splits
    train_dataset, valid_dataset, test_dataset, dataset = load_dataset(train_args)
    
    # Train ensemble members
    ensemble_paths = []
    for i in range(n_ensemble):
        member_path = train_single_ensemble_member(
            i, deepcopy(model_args), deepcopy(train_args), 
            train_dataset, valid_dataset, test_dataset, dataset, path
        )
        ensemble_paths.append(member_path)
        print(f"Completed ensemble member {i+1}/{n_ensemble}")
    
    # Perform ensemble uncertainty estimation
    ensemble_results = ensemble_uncertainty_estimation(ensemble_paths, path)
    
    print("Ensemble training completed!")
    return path, ensemble_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--submitit','-s', type=bool, default=False)
    parser.add_argument('--n_ensemble', type=int, default=5, help='Number of ensemble members')
    parser.add_argument('--test', type=bool, default=False)
    
    args = parser.parse_args()
    
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    model_args = config_dict['model_args']
    train_args = config_dict['train_args']
    
    # Add base seed for ensemble if not present
    if 'base_seed' not in train_args:
        train_args['base_seed'] = train_args['seed']
    
    train_args['test'] = args.test
    train_args['optimizer_args']['lr'] = train_args['learning_rate']
    
    original_train_args = deepcopy(train_args)
    ppath = Path("./runs", time.strftime("%m-%d"),
                   f"{train_args['dataset']}_ensemble_{model_args['name']}_loss{train_args['loss_fn']}_kl{train_args['kl_weight']:.0E}_batch{train_args['loader_tr_args']['batch_size']}_lr{train_args['learning_rate']:.0E}_{time.strftime('%H-%M')}")
    ppath.mkdir(parents=True, exist_ok=True)
    with open(Path(ppath, "results.csv"), 'w') as f:
        f.write("seed, dp_mae, dp_l2dist,dp_miscalibration_area, dp_calibration_score\n")

    # === Define the run function ===
    def run_one_ensemble_member(seed):
        this_args = deepcopy(original_train_args)
        this_args['seed'] = seed
        this_args['base_seed'] = seed
        ensemble_path, results = ensemble_training(model_args, this_args, ppath, args.n_ensemble)
        for key, value in results.items():
            print(f"{key}: {value:.4f}")

    # === Submitit logic ===
    if args.submitit and len(train_args['seed'])  > 1:
        import submitit
        executor = submitit.AutoExecutor(folder=ppath)
        executor.update_parameters(
            name=f"{train_args['dataset']}_{model_args['name']}_ens",
            mem_gb=40,
            timeout_min=600,
            slurm_partition="a100",
            gpus_per_node=1,
            cpus_per_task=4,
            tasks_per_node=1,
            nodes=1,
            slurm_exclude="p2-gpu-15", #3-gpu-[43-50]"
        )
        jobs = executor.map_array(run_one_ensemble_member, list(original_train_args['seed']))
        print("Jobs submitted:", [job.job_id for job in jobs])
        sys.exit(0)

    # === Single job (Submitit child or local run) ===
    if args.submitit and len(train_args['seed']) == 1:
        run_one_ensemble_member(train_args['seed'][0])
        print("Single job run completed.")
        sys.exit(0)

    # === Local non-submitit run ===
    original_train_args = deepcopy(train_args)
    for seed in original_train_args['seed']:
        train_args = deepcopy(original_train_args)
        train_args['seed'] = seed
        train_args['base_seed'] = seed
        ensemble_path, results = ensemble_training(model_args, train_args, ppath, args.n_ensemble)
    
        print(f"Seed {seed} done.")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")