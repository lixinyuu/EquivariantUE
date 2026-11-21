import os
import sys

HOST=os.uname().nodename
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

from uncertainty_ai.dataset import MD17, QM7X, FloatToTensorTransform, QM9P, MD17Aligned, RotatedDataset
from uncertainty_ai.metrics.calibration_scores import calibration_regression, nll
from uncertainty_ai.metrics.force_calibration import (energy_calibration,
                                                        calibration_regression_vector,
                                                        miscalibration_area_vector,
                                                        get_force_cov)
from uncertainty_ai.metrics.uct_utils import write_all_metrics
from training_lightning import (LightningTraining, 
                                LightningTrainingEvidential, 
                                LightningTrainingForceEvidential,
                                LightningTrainingForceEvidentialTwoStage,
                                LightningTrainingDropout, 
                                LightningTrainingVariationalDropout)
from uncertainty_ai.utils import get_model_args, has_tensor_cores
from ocpmodels.datasets import LmdbDataset

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger


RESULTS_COLUMNS = [
    "seed",
    "energy_mae",
    "force_mae",
    "energy_miscalibration_area",
    "energy_calibration_score",
    "force_miscalibration_area",
    "force_calibration_score",
    "energy_sharpness",
    "x_dimwise_calib",
    "y_dimwise_calib",
    "z_dimwise_calib",
    "max_dimwise_calib",
    "x_dimwise_nll",
    "y_dimwise_nll",
    "z_dimwise_nll",
    "max_dimwise_nll",
]



def set_python_path():
    import sys
    import os
    HOST=os.uname().nodename
    sys.path.insert(0, CURRENT_DIR)
    

@rank_zero_only
def collect_and_write_metrics(world_size, results_path, student_dof=None):
    print("Collecting and writing metrics using rank 0")
    e_pred, e_true, f_pred, f_true, e_stds, f_stds = [], [], [], [], [], []
    for i in range(world_size):
        e_pred.append(torch.load(Path(results_path, f"energy_pred_{i}.pt")))
        e_true.append(torch.load(Path(results_path, f"energy_true_{i}.pt")))
        f_pred.append(torch.load(Path(results_path, f"force_pred_{i}.pt")))
        f_true.append(torch.load(Path(results_path, f"force_true_{i}.pt")))
        e_stds.append(torch.load(Path(results_path, f"energy_stds_{i}.pt")))
        f_stds.append(torch.load(Path(results_path, f"force_stds_{i}.pt")))
    
    # Load per-sample nu if saved (ForceEvi)
    has_nu = all(
        Path(results_path, f"force_nu_{i}.pt").exists()
        and Path(results_path, f"force_kappa_{i}.pt").exists()
        for i in range(world_size)
    )
    if has_nu:
        f_nu_list = []
        f_kappa_list = []
        for i in range(world_size):
            f_nu_list.append(torch.load(Path(results_path, f"force_nu_{i}.pt")))
            f_kappa_list.append(torch.load(Path(results_path, f"force_kappa_{i}.pt")))
        f_nu_all = torch.cat(f_nu_list, dim=0)
        f_kappa_all = torch.cat(f_kappa_list, dim=0)
        student_dof = f_nu_all  # per-sample tensor
    
    e_pred = torch.cat(e_pred, dim=0)
    e_true = torch.cat(e_true, dim=0)
    f_pred = torch.cat(f_pred, dim=0)
    f_true = torch.cat(f_true, dim=0)
    e_stds = torch.cat(e_stds, dim=0)
    f_stds = torch.cat(f_stds, dim=0)

    e_pred = e_pred.flatten()
    e_true = e_true.flatten()
    e_stds = e_stds.flatten()

    e_stds[e_stds < 1e-6] = 1e-6
    if f_stds.dim() == 3:
        f_stds = f_stds + 1e-6 * torch.eye(f_stds.shape[-1], dtype=f_stds.dtype)
    else:
        f_stds[f_stds < 1e-6] = 1e-6

    e_mae = float(torch.nn.functional.l1_loss(e_pred, e_true))
    f_mae = float(torch.nn.functional.l1_loss(f_pred, f_true))

    uct_results = write_all_metrics(e_pred.numpy(), e_stds.numpy(), e_true.numpy(), output_file=Path(results_path, "energy_metrics.txt").as_posix())
    e_miscalibration_area = uct_results['avg_calibration']['miscal_area']
    if f_pred.shape[0] == f_stds.shape[0]:
        # if f_stds is 1D
        if len(f_stds.shape) == 1:
            # repeat 3 times in the last dimension
            f_stds = f_stds.repeat(1, 3)
            print("f_stds has been repeated 3 times in the last dimension")
            print("results not guaranteed")
        elif f_stds.shape[1] == 1:
            # repeat 3 times in the last dimension
            f_stds = f_stds.repeat(1, 3)
            print("f_stds has been repeated 3 times in the last dimension")
            print("results not guaranteed")
    
    if len(f_stds.shape) == 3:
        # For ForceEvi with per-sample nu: use F-distribution threshold
        # f_stds = alea + epis = Psi/df*(1 + 1/kappa) = (kappa+1)/(kappa*df)*Psi
        # This IS the Student-t scale matrix Sigma_t, so use it directly.
        # df = nu - d + 1 (d=3) is the Student-t degrees of freedom.
        if has_nu and student_dof is not None:
            d = 3
            df = torch.clamp(f_nu_all - d + 1.0, min=1e-6)  # [B]
            f_miscalibration_area = miscalibration_area_vector(
                f_pred, f_stds, f_true,
                student_dof=df, pred_cov_is_covariance=False,
            )
            f_calibration_score = calibration_regression_vector(
                f_pred, f_stds, f_true,
                student_dof=df, pred_cov_is_covariance=False,
            )
        else:
            f_miscalibration_area = miscalibration_area_vector(f_pred, f_stds, f_true)
            f_calibration_score = calibration_regression_vector(f_pred, f_stds, f_true)
    else:
        f_miscalibration_area = energy_calibration(f_pred.numpy().flatten(), f_true.numpy().flatten(), f_stds.numpy().flatten())
        f_calibration_score = calibration_regression(f_pred.flatten(), f_stds.flatten(), f_true.flatten())
    e_sharpness = uct_results['sharpness']['sharp']

    # Dimension-wise calibration and NLL from diagonal of covariance
    f_dimwise_calib = float('nan')
    f_dimwise_nll = float('nan')
    f_max_calib = float('nan')
    f_max_nll = float('nan')
    dim_calibs = [float('nan')] * 3
    dim_nlls = [float('nan')] * 3
    if len(f_stds.shape) == 3:
        import torch.distributions as D
        f_diag_stds = torch.sqrt(torch.diagonal(f_stds, dim1=-2, dim2=-1))  # [B, 3]
        # Max calibration score across 3 dimensions
        for dim_i in range(3):
            dim_calibs[dim_i] = calibration_regression(
                f_pred[:, dim_i], f_diag_stds[:, dim_i], f_true[:, dim_i]
            )
            dim_nlls[dim_i] = float(nll(D.Normal(f_pred[:, dim_i], f_diag_stds[:, dim_i]), f_true[:, dim_i]))

        f_max_calib = max(dim_calibs)
        f_max_nll = max(dim_nlls)

    print("energy mae = {:.4f} force mae = {:.4f}".format(e_mae, f_mae))
    print("energy miscalibration_area = {:.4f}".format(e_miscalibration_area))
    print("energy sharpness = {:.4f}".format(e_sharpness))
    
    print("energy calibration score = {:.4f}".format(calibration_regression(e_pred, e_stds, e_true)))
    print("force miscalibration_area = {:.4f}".format(f_miscalibration_area))
    print("force calibration score = {:.4f}".format(f_calibration_score))
    with open(Path(results_path, "results.txt") ,'w') as fp:
        fp.write("energy mae = {:.4f} force mae = {:.4f}\n".format(e_mae, f_mae))
        fp.write("energy miscalibration_area = {:.4f}\n".format(e_miscalibration_area))
        fp.write("energy sharpness = {:.4f}\n".format(e_sharpness))
        fp.write("force miscalibration_area = {:.4f}\n".format(f_miscalibration_area))
        fp.write("energy calibration score = {:.4f}\n".format(calibration_regression(e_pred, e_stds, e_true)))
        fp.write("force calibration score = {:.4f}\n".format(f_calibration_score))
        fp.write("force dimwise nll = {:.4f}\n".format(f_max_nll))
        fp.write("force max calibration = {:.4f}\n".format(f_max_calib))
    dir_path = Path(results_path).parent
    seed = Path(results_path).name
    with open(Path(dir_path, "results.txt"), 'a') as f:
        f.write(f"{seed},{e_mae},{f_mae},{e_miscalibration_area},{calibration_regression(e_pred,e_stds,e_true)},{f_miscalibration_area},{f_calibration_score},{e_sharpness},{dim_calibs[0]},{dim_calibs[1]},{dim_calibs[2]},{f_max_calib},{dim_nlls[0]},{dim_nlls[1]},{dim_nlls[2]},{f_max_nll}\n")


def load_results_table(path):
    csv = pd.read_csv(Path(path, "results.txt"), names=RESULTS_COLUMNS, skiprows=1)
    csv["seed_numeric"] = pd.to_numeric(csv["seed"], errors="coerce")
    csv = csv[csv["seed_numeric"].notna()].drop(columns=["seed_numeric"])
    csv = csv.set_index("seed")
    return csv.apply(pd.to_numeric, errors="coerce")

@rank_zero_only
def results_analysis(path):
    csv = load_results_table(path)
    mean_values = csv.values.mean(axis=0)
    std_values = csv.values.std(axis=0)

    with open(Path(path, "results.txt"), 'a') as f:
        f.write("mean, "+ ", ".join([str(i) for i in mean_values]) + "\n")
        f.write("std, "+ ", ".join([str(i) for i in std_values]) + "\n")

def results_analysis_global(path):
    csv = load_results_table(path)
    mean_values = csv.values.mean(axis=0)
    std_values = csv.values.std(axis=0)

    with open(Path(path, "results.txt"), 'a') as f:
        f.write("mean, "+ ", ".join([str(i) for i in mean_values]) + "\n")
        f.write("std, "+ ", ".join([str(i) for i in std_values]) + "\n")


def set_seeds(seed:int):
    """Set seed

    Args:
        seed (int): seed
    
    Returns:
        None
    """
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
        

def independent_running(seed, model_args = {}, train_args = {}):
    path = Path(train_args['path'], str(seed))
    path.mkdir(parents=True, exist_ok=True )
    writer = TensorBoardLogger(save_dir = path)
    csv_logger = CSVLogger(save_dir=path, name='csv')
    # set seed
    set_seeds(seed)
    set_python_path()

    # set tensor cores
    if has_tensor_cores():
        torch.set_float32_matmul_precision('high')
        print("Using tensor cores")



    # load dataset
    if train_args['dataset'] == 'qm7x':
        dataset = QM7X("./dataset/QM7X")
        train_dataset, valid_dataset, test_dataset = random_split(dataset, [33213, 4151, 4153])
    elif train_args['dataset'] == "qm9":
        dataset = QM9P(root = "./dataset/QM9" )
        train_dataset, valid_dataset, test_dataset = random_split(dataset, [100000, 10000, len(dataset) - 110000])
        # Debug
    elif train_args['dataset'] == 'oc20':
        train_dataset = LmdbDataset({"src": Path(CURRENT_DIR, "dataset/s2ef-200k/train_new")}, transform = FloatToTensorTransform())
        dataset = LmdbDataset({"src": Path(CURRENT_DIR, "dataset/s2ef-200k/val_id_new")}, transform = FloatToTensorTransform())
        valid_dataset, test_dataset, _ = random_split(dataset, [2000, 2000, len(dataset) - 4000])
    elif train_args['dataset'] == 'oc20_100k':
        train_dataset = LmdbDataset({"src": Path(CURRENT_DIR, "dataset/s2ef-200k/train_new")}, transform = FloatToTensorTransform())
        dataset = LmdbDataset({"src": Path(CURRENT_DIR, "dataset/s2ef-200k/val_id_new")}, transform = FloatToTensorTransform())
        train_dataset = Subset(train_dataset, np.arange(100000)*2)
        valid_dataset, test_dataset, _ = random_split(dataset, [2000, 2000, len(dataset) - 4000])
    elif "ood" in train_args['dataset']:
        dataset = MD17Aligned("./dataset", train_args['dataset'][:-4]) # remove the "_ood" suffix to get the original dataset name
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        idx = idx[:4000]
        dataset = Subset(dataset, idx)
        train_dataset, valid_dataset, _test_dataset = random_split(dataset, [1000, 1000, len(dataset)-2000])
        test_dataset = RotatedDataset(_test_dataset)
    else:
        dataset = MD17("./dataset", train_args['dataset'])
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        idx = idx[:4000]
        dataset = Subset(dataset, idx)
        train_dataset, valid_dataset, test_dataset = random_split(dataset, [1000, 1000, len(dataset)-2000])
    
    if train_args['test']:
        train_dataset = Subset(train_dataset, np.arange(500))
        valid_dataset = Subset(valid_dataset, np.arange(500))
        test_dataset = Subset(test_dataset, np.arange(500))
        train_args['n_epoch'] = 10 if train_args['n_epoch'] > 10 else train_args['n_epoch']
            
        
    
    model_name = model_args['name']

    ### Get dataset energy mean by sampling 5000 samples
    if len(train_dataset) < 5000:
        energy_mean = np.mean([d.y.numpy() for d in train_dataset])
        energy_std = np.std([d.y.numpy() for d in train_dataset])
    else:
        sample_idx = np.random.choice(len(train_dataset), 5000, replace=False)
        energy_mean = np.mean([train_dataset[i].y.numpy() for i in sample_idx])
        energy_std = np.std([train_dataset[i].y.numpy() for i in sample_idx])
    print("Energy mean = {:.4f}, Energy std = {:.4f}".format(energy_mean, energy_std))
    

    
    if train_args['normalizer']:
        train_args['normalizer_args']['mean'] = energy_mean
        print("Automatically set energy mean to {:.4f}".format(energy_mean))
    
    # MACE have a arg of elements_type, which should be obtained from the dataset
    if model_args['name'] == 'MACE2':
        atomic_numbers= dataset[0].atomic_numbers
        model_args['elements_type'] = torch.unique(atomic_numbers).tolist()
        model_args['atomic_energies'] = np.zeros(len(model_args['elements_type']))
        print("Automatically set elements_type to {}".format(model_args['elements_type']))
        # Now fit the atomic energies
        X = []
        y = []
        for d in train_dataset:
            atomic_numbers = d.atomic_numbers.numpy()
            element_counts = np.bincount(atomic_numbers)
            element_counts = element_counts[model_args['elements_type']]
            X.append(element_counts)
            y.append((d.y.numpy()-train_args['normalizer_args']['mean'])/train_args['normalizer_args']['std'])
        X = np.array(X)
        y = np.array(y)
        # Regularization strength
        alpha = 0.1

        # Identity matrix for ridge penalty
        I = np.eye(X.shape[1])

        # Closed-form solution
        w = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
        
        
    temp_model_args = deepcopy(model_args)
    del temp_model_args['name']
    gnn_model, model_args = get_model_args(model_name, temp_model_args)

    save_args(model_args, path, 'model_args')
    save_args(train_args, path, 'train_args')
    train_args['sub_path'] = path
    device = 'cuda'
    if "EnergyEvi" in train_args['loss_fn']:
        model = LightningTrainingEvidential(gnn_model, model_args, train_args, writer, device, amp = train_args['amp'], init_model=True)
    elif "ForceEvi" in train_args['loss_fn']:
        # Check if we should use two-stage training
        model = LightningTrainingForceEvidential(gnn_model, model_args, train_args, writer, device, amp = train_args['amp'], init_model=True)
    elif "VBDDropout" in model_name or "VariationalDropout" in model_name:
        model = LightningTrainingVariationalDropout(gnn_model, model_args, train_args, writer, device, amp = train_args['amp'], init_model=True)
    elif "Drop" in model_name:
        model = LightningTrainingDropout(gnn_model, model_args, train_args, writer, device, amp = train_args['amp'], init_model=True)
    else:
        model = LightningTraining(gnn_model, model_args, train_args, writer, device, amp = train_args['amp'], init_model=True)
    # train model# train model
    if 'pretrained' in train_args and train_args['pretrained']:
        model.load_state_dict(torch.load(train_args['pretrained']))
        print("Load pretrained model from ", train_args['pretrained'])

    trainer = pl.Trainer(accelerator="cuda", 
                         max_epochs=train_args['n_epoch'], 
                         devices=torch.cuda.device_count(), 
                         inference_mode=False, 
                         precision="bf16-mixed" if train_args['amp'] else "32",
                         logger = [writer, csv_logger])
    trainer.fit(model, DataLoader(train_dataset, **train_args['loader_tr_args']), DataLoader(valid_dataset, **train_args['loader_te_args']))
    f_pred = []
    f_true = []
    e_pred = []
    e_true = []
    e_stds = []
    f_stds = []
    if "ForceEvi" in train_args['loss_fn']:
        test_loader = DataLoader(test_dataset, **train_args['loader_te_args'])
        predictions = trainer.predict(model, test_loader)
        
        f_nu = []
        f_kappa = []
        for idx, pred in enumerate(predictions):
            e_pred_, e_true_, f_pred_, f_true_, f_stds_, kappa_, nu_ = pred
            e_pred.append(e_pred_.detach().to("cpu"))
            e_true.append(e_true_.detach().to("cpu"))
            f_pred.append(f_pred_.detach().to("cpu"))
            f_true.append(f_true_.detach().to("cpu"))
            f_stds.append(f_stds_.detach().to("cpu"))
            f_nu.append(nu_.detach().to("cpu"))
            f_kappa.append(kappa_.detach().to("cpu"))

        print("Evidential test, generate random energy stds")
        e_stds = [torch.rand(e_pred_.size(0)) for e_pred_ in e_pred]
        f_nu = torch.cat(f_nu, dim=0).flatten()
        f_kappa = torch.cat(f_kappa, dim=0).flatten()
    elif "Drop" in model_name:
        test_loader = DataLoader(test_dataset, **train_args['loader_te_args'])
        predictions = trainer.predict(model, test_loader)

        for idx, pred in enumerate(predictions):
            e_pred_, e_true_, f_pred_, f_true_, e_stds_, f_stds_ = pred
            e_pred.append(e_pred_.detach().to("cpu"))
            e_true.append(e_true_.detach().to("cpu"))
            f_pred.append(f_pred_.detach().to("cpu"))
            f_true.append(f_true_.detach().to("cpu"))
            e_stds.append(e_stds_.detach().to("cpu"))
            f_stds.append(f_stds_.detach().to("cpu"))

    elif "Evi" in train_args['loss_fn']:
        test_loader = DataLoader(test_dataset, **train_args['loader_te_args'])
        predictions = trainer.predict(model, test_loader)
        
        for idx, pred in enumerate(predictions):
            e_pred_, e_true_, f_pred_, f_true_, e_stds_ = pred
            e_pred.append(e_pred_.detach().to("cpu"))
            e_true.append(e_true_.detach().to("cpu"))
            f_pred.append(f_pred_.detach().to("cpu"))
            f_true.append(f_true_.detach().to("cpu"))
            e_stds.append(e_stds_.detach().to("cpu"))

        print("Evidential test, generate random force stds")
        f_stds = [torch.eye(3)[None, :, :].repeat(f_pred_.size(0), 1, 1) for f_pred_ in f_pred]

    else:
        test_loader = DataLoader(test_dataset, **train_args['loader_te_args'])
        predictions = trainer.predict(model, test_loader)
        
        for idx, pred in enumerate(predictions):
            e_pred_, e_true_, f_pred_, f_true_ = pred
            e_pred.append(e_pred_.detach().to("cpu"))
            e_true.append(e_true_.detach().to("cpu"))
            f_pred.append(f_pred_.detach().to("cpu"))
            f_true.append(f_true_.detach().to("cpu"))

        print("Standard test, generate random energy and force stds")
        e_stds = [torch.rand(e_pred_.size(0)) for e_pred_ in e_pred]
        f_stds = [torch.eye(3).view(-1, 3, 3).repeat(f_pred_.size(0), 1, 1) for f_pred_ in f_pred]

    # if it is mupltiple gpu, get the current rank
    if trainer.world_size > 1:
        rank = trainer.global_rank
        world_size = trainer.world_size
    else:
        rank = 0
        world_size = 1
    
    # concatenate the results to a single torch tensor
    e_pred = torch.cat(e_pred, dim=0).flatten()
    e_true = torch.cat(e_true, dim=0).flatten()
    f_pred = torch.cat(f_pred, dim=0)
    f_true = torch.cat(f_true, dim=0)
    e_stds = torch.cat(e_stds, dim=0).flatten()
    f_stds = torch.cat(f_stds, dim=0)

    torch.save(e_pred, Path(path, f"energy_pred_{rank}.pt"))
    torch.save(e_true, Path(path, f"energy_true_{rank}.pt"))
    torch.save(f_pred, Path(path, f"force_pred_{rank}.pt"))
    torch.save(f_true, Path(path, f"force_true_{rank}.pt"))
    torch.save(e_stds, Path(path, f"energy_stds_{rank}.pt"))
    torch.save(f_stds, Path(path, f"force_stds_{rank}.pt"))

    # Save per-sample nu and kappa for ForceEvi (needed for DDP gather)
    if "ForceEvi" in train_args['loss_fn']:
        torch.save(f_nu, Path(path, f"force_nu_{rank}.pt"))
        torch.save(f_kappa, Path(path, f"force_kappa_{rank}.pt"))

    e_mae = float(torch.nn.functional.l1_loss(e_pred, e_true))
    f_mae = float(torch.nn.functional.l1_loss(f_pred, f_true))
    print("energy mae = {:.4f} force mae = {:.4f}".format(e_mae, f_mae))

    # if it is ddp mode, use barrier to synchronize
    if world_size > 1:
        trainer.strategy.barrier()
        print("sleep 10 seconds to wait for all processes to finish")
        time.sleep(10)
        # ! not the follwing part did not process correctly
        collect_and_write_metrics(world_size, path)
        trainer.strategy.barrier()
        print("sleep 5 seconds to wait for all processes to finish")
        time.sleep(5)
    else:
        collect_and_write_metrics(world_size, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='?', type=argparse.FileType('r'),
                    default=sys.stdin)
    parser.add_argument('--parallel','-p', type=bool, default=False)
    parser.add_argument('--submitit','-s', type=bool, default=False)
    parser.add_argument('--test',type=bool, default=False)
    args = parser.parse_args()
    if args.parallel:
        set_start_method('spawn', force=True)
        
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    model_args=config_dict['model_args']
    train_args=config_dict['train_args']
    if "ForceEvi" in train_args['loss_fn']:
        if "use_alpha" in model_args:
            path = Path("./runs" , time.strftime("%m-%d"), train_args['dataset'] + "_ForceEvi_"+ model_args['name'] + "_alpha" + str(model_args["use_alpha"]) + "_l_dim" + str(model_args["l_dim"])
            + '_loss' + str(train_args['loss_fn'])+ "{:.1E}".format(train_args["loss_fn_args"]["rho_force"]) + '_batch' + str(train_args['loader_tr_args']['batch_size']) +'_lr' 
            + "{:.0E}".format(train_args['learning_rate']) + "_" + time.strftime("%H-%M"))
        else:
            path = Path("./runs" , time.strftime("%m-%d"), train_args['dataset'] + "_ForceEvi_"+ model_args['name'] 
            + '_loss' + str(train_args['loss_fn']) + "{:.1E}".format(train_args["loss_fn_args"]["rho_force"])+ '_batch' + str(train_args['loader_tr_args']['batch_size']) +'_lr' 
            + "{:.0E}".format(train_args['learning_rate']) + "_" + time.strftime("%H-%M"))
    elif ("Dropout" in model_args['name']) and "VBDDropout" not in model_args['name']:
        path = Path("./runs" , time.strftime("%m-%d"), train_args['dataset'] + "_"+ model_args['name'] 
            + '_type' + str(model_args['dropout_method']) +'_ratio'+"{:.0E}".format(model_args['dropout_p'])+ '_batch' + str(train_args['loader_tr_args']['batch_size']) +'_lr' 
            + "{:.0E}".format(train_args['learning_rate']) + "_" + time.strftime("%H-%M"))
    elif "VBDDropout" in model_args['name']:
        path = Path("./runs" , time.strftime("%m-%d"), train_args['dataset'] + "_"+ model_args['name'] 
            + '_method' + str(model_args['vbd_method']) +'_kl'+"{:.0E}".format(train_args['kl_weight'])+ '_batch' + str(train_args['loader_tr_args']['batch_size']) +'_lr' 
            + "{:.0E}".format(train_args['learning_rate']) + "_" + time.strftime("%H-%M"))
    else:
        path = Path("./runs" , time.strftime("%m-%d"), train_args['dataset'] + "_"+ model_args['name'] 
            + '_loss' + str(train_args['loss_fn']) +'_kl'+"{:.0E}".format(train_args['kl_weight'])+ '_batch' + str(train_args['loader_tr_args']['batch_size']) +'_lr' 
            + "{:.0E}".format(train_args['learning_rate']) + "_" + time.strftime("%H-%M"))
    path.mkdir(parents=True, exist_ok=True )
    train_args['path'] = path
    train_args['test'] = args.test
    train_args['optimizer_args']['lr'] = train_args['learning_rate']

    


    with open(Path(path, "results.txt"), 'w') as f:
        f.write(",".join(RESULTS_COLUMNS) + "\n")

    if args.parallel:
        with Pool() as pool:
            N = pool.map(partial(independent_running, model_args = model_args, train_args = train_args), train_args['seed'])
        results_analysis(path)
    elif args.submitit:
        def independent_running_helper(seed):
            independent_running(seed, model_args = model_args, train_args = train_args)
            
        import submitit
        from submitit.helpers import DelayedSubmission
        executor = submitit.AutoExecutor(folder=path)

        executor2 = submitit.AutoExecutor(folder=path)
        print(executor.cluster)
        if executor.cluster == "slurm":
            print("Executor will schedule jobs on Slurm.")
            executor.update_parameters(
                name=str(train_args['dataset']) + '_' + str(model_args['name']),
                # account="strategic",
                mem_gb=50,
                timeout_min=int(5*60),
                slurm_partition="a100",
                gpus_per_node=1,
                cpus_per_task=4,
                tasks_per_node=1,
                nodes=1,
                slurm_exclude="p3-gpu-48", #3-gpu-[43-50]"
                # slurm_additional_parameters=,
            
            )
            jobs = executor.map_array(independent_running_helper, train_args['seed'])
            print(jobs[-1].job_id, " submited, now we submit the analysis job")
            executor2.update_parameters(
                name=str(train_args['dataset']) + '_' + str(model_args['name'])+"_Ana",
                mem_gb=5,
                timeout_min=10,
                slurm_partition="a100cpu",
                cpus_per_task=1,
                tasks_per_node=1,
                nodes=1,
                slurm_additional_parameters={"dependency": "afterok:"+",".join([str(jobs[-1].job_id) for job in jobs])},
            )
            jobs2 = executor2.submit(results_analysis_global, path)
            print(f"Analysis job {jobs2.job_id} submited, exit now")
        else:
            print(f"!!! Slurm executable `srun` not found. Will execute jobs on local machine. !!!")
            for seed in train_args.seeds:
                independent_running(seed, model_args = model_args, train_args = train_args)
    else:
        for seed in train_args['seed']:
            independent_running(seed, model_args = model_args, train_args = train_args)
        results_analysis(path)


