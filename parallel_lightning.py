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
import pickle
from torch.multiprocessing import Pool, set_start_method
from torch.utils.data import Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from uncertainty_ai.dataset import (
    MD17,
    QM7X,
    QM7xDPAligned,
    DPRotatedDataset,
    FloatToTensorTransform,
    QM9P,
)
from uncertainty_ai.metrics.calibration_scores import calibration_regression, nll
from uncertainty_ai.metrics.force_calibration import (energy_recalibration,
                                                        calibration_regression_vector,
                                                        miscalibration_area_vector,
                                                        get_force_cov,
                                                        plot_calibration)
from uncertainty_ai.metrics.uct_utils import write_all_metrics
from training_lightning import LightningTraining, LightningTrainingForceEvidential, LightningTrainingDropout, LightningTrainingVariationalDropout
from uncertainty_ai.utils import get_model_args, has_tensor_cores
from ocpmodels.datasets import LmdbDataset

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

RESULTS_COLUMNS = [
    "seed",
    "dp_mae",
    "dp_l2dist",
    "dp_miscalibration_area",
    "dp_calibration_score",
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
    sys.path.insert(0, CURRENT_DIR)

@rank_zero_only
def collect_and_write_metrics(world_size, results_path):
    print("Collecting and writing metrics using rank 0")
    e_pred, e_true, dp_pred, dp_true, e_stds, dp_stds = [], [], [], [], [], []
    for i in range(world_size):
        dp_pred.append(torch.load(Path(results_path, f"dp_pred_{i}.pt")))
        dp_true.append(torch.load(Path(results_path, f"dp_true_{i}.pt")))
        dp_stds.append(torch.load(Path(results_path, f"dp_stds_{i}.pt")))
    
    dp_pred = torch.cat(dp_pred, dim=0)
    dp_true = torch.cat(dp_true, dim=0)
    dp_stds = torch.cat(dp_stds, dim=0)

    
    

    if dp_pred.shape[0] == dp_stds.shape[0]:
        # if f_stds is 1D
        if len(dp_stds.shape) == 1:
            # repeat 3 times in the last dimension
            dp_stds = dp_stds.unsqueeze(-1).repeat(1, 3)
            print("f_stds has been repeated 3 times in the last dimension")
            print("results not guaranteed")
        elif dp_stds.shape[1] == 1:
            # repeat 3 times in the last dimension
            dp_stds = dp_stds.repeat(1, 3)
            print("f_stds has been repeated 3 times in the last dimension")
            print("results not guaranteed")
    if len(dp_pred.shape) == 3:
        # if it is sampling output, take the mean
        dp_pred = dp_pred.mean(dim=1)
        print("dp_pred has shape {}, take the mean".format(dp_pred.shape))
    f_mae = float(torch.nn.functional.l1_loss(dp_pred.norm(dim=-1), dp_true.norm(dim=-1)))
    f_l2dist = float((dp_pred - dp_true).norm(dim=-1).mean().item())
    f_miscalibration_area = miscalibration_area_vector(dp_pred, dp_stds, dp_true)
    f_calibration_score = calibration_regression_vector(dp_pred, dp_stds, dp_true)

    if dp_stds.ndim == 3:
        dp_dimwise_stds = torch.sqrt(torch.clamp(torch.diagonal(dp_stds, dim1=-2, dim2=-1), min=1e-12))
    else:
        dp_dimwise_stds = torch.clamp(dp_stds, min=1e-6)

    dim_calibs = []
    dim_nlls = []
    for dim_i in range(dp_pred.shape[-1]):
        dim_calibs.append(
            float(calibration_regression(dp_pred[:, dim_i], dp_dimwise_stds[:, dim_i], dp_true[:, dim_i]))
        )
        dim_nlls.append(
            float(nll(dp_pred[:, dim_i], dp_dimwise_stds[:, dim_i], dp_true[:, dim_i]))
        )

    max_dimwise_calib = max(dim_calibs)
    max_dimwise_nll = max(dim_nlls)
    
    print("dp mae = {:.4f}".format(f_mae))
    print("dp miscalibration_area = {:.4f}".format(f_miscalibration_area))
    print("dp calibration score = {:.4f}".format(f_calibration_score))
    print(
        "dp dimwise calibration x/y/z = {:.4f}/{:.4f}/{:.4f} max = {:.4f}".format(
            dim_calibs[0], dim_calibs[1], dim_calibs[2], max_dimwise_calib
        )
    )
    print(
        "dp dimwise nll x/y/z = {:.4f}/{:.4f}/{:.4f} max = {:.4f}".format(
            dim_nlls[0], dim_nlls[1], dim_nlls[2], max_dimwise_nll
        )
    )

    ax = plot_calibration(dp_pred, dp_stds, dp_true)
    plt.savefig(Path(results_path, "calibration_plot.png"))
    plt.close(ax.figure)
    with open(Path(results_path, "results.txt") ,'w') as fp:
        fp.write("dp mae = {:.4f}\n".format(f_mae))
        fp.write("dp miscalibration_area = {:.4f}\n".format(f_miscalibration_area))
        fp.write("dp calibration score = {:.4f}\n".format(f_calibration_score))
        fp.write(
            "dp dimwise calibration x/y/z = {:.4f}/{:.4f}/{:.4f} max = {:.4f}\n".format(
                dim_calibs[0], dim_calibs[1], dim_calibs[2], max_dimwise_calib
            )
        )
        fp.write(
            "dp dimwise nll x/y/z = {:.4f}/{:.4f}/{:.4f} max = {:.4f}\n".format(
                dim_nlls[0], dim_nlls[1], dim_nlls[2], max_dimwise_nll
            )
        )
    dir_path = Path(results_path).parent
    seed = Path(results_path).name
    with open(Path(dir_path, "results.csv"), 'a') as f:
        f.write(
            f"{seed},{f_mae},{f_l2dist},{f_miscalibration_area},{f_calibration_score},"
            f"{dim_calibs[0]},{dim_calibs[1]},{dim_calibs[2]},{max_dimwise_calib},"
            f"{dim_nlls[0]},{dim_nlls[1]},{dim_nlls[2]},{max_dimwise_nll}\n"
        )


def load_results_table(path):
    csv = pd.read_csv(Path(path, "results.csv"), names=RESULTS_COLUMNS, skiprows=1)
    csv["seed_numeric"] = pd.to_numeric(csv["seed"], errors="coerce")
    csv = csv[csv["seed_numeric"].notna()].drop(columns=["seed_numeric"])
    csv = csv.set_index("seed")
    return csv.apply(pd.to_numeric, errors="coerce")

@rank_zero_only
def results_analysis(path):
    csv = load_results_table(path)
    mean_values = csv.values.mean(axis=0)
    std_values = csv.values.std(axis=0)

    with open(Path(path, "results.csv"), 'a') as f:
        f.write("mean, "+ ", ".join([str(i) for i in mean_values]) + "\n")
        f.write("std, "+ ", ".join([str(i) for i in std_values]) + "\n")

def results_analysis_global(path):
    csv = load_results_table(path)
    mean_values = csv.values.mean(axis=0)
    std_values = csv.values.std(axis=0)

    with open(Path(path, "results.csv"), 'a') as f:
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
        dataset = QM7X(
            "./dataset/QM7X",
            target=train_args['dataset_target'],
            minima="global",
        )
    elif train_args['dataset'] == 'qm7x_ood':
        dataset = QM7xDPAligned(
            "./dataset/QM7X",
            target=train_args['dataset_target'],
            minima="global",
            plane=(1.0, 1.0, 0.0),
        )
    else:
        raise ValueError(f"Unsupported dataset for this script: {train_args['dataset']}")

    #train_dataset, valid_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    # Use index to split the dataset therefore inddex can be saved
    n_data = len(dataset)
    train_idx, val_idx, test_idx = random_split(np.arange(n_data), [0.8, 0.1, 0.1]
                                               , generator=torch.Generator().manual_seed(seed))
    # save the index to path
    with open(Path(path, "split.pkl"), 'wb') as f:
        pickle.dump((train_idx, val_idx, test_idx), f)

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, val_idx)
    if train_args['dataset'] == 'qm7x_ood':
        test_dataset = DPRotatedDataset(Subset(dataset, test_idx))
    else:
        test_dataset = Subset(dataset, test_idx)
    print("Train dataset size: ", len(train_dataset))
    print(train_dataset[0])
    
    model_name = model_args['name']

    ### Get dataset energy mean by sampling 5000 samples
    train_args['normalizer_args']['mean'] = 0.0
    train_args['normalizer_args']['std'] = 1.0

        
    temp_model_args = deepcopy(model_args)
    del temp_model_args['name']
    gnn_model, model_args = get_model_args(model_name, temp_model_args)

    save_args(model_args, path, 'model_args')
    save_args(train_args, path, 'train_args')
    train_args['sub_path'] = path
    device = 'cuda'
    if "Evi" in train_args['loss_fn']:
        model = LightningTrainingForceEvidential(gnn_model, model_args, train_args, writer, device, amp = train_args['amp'], init_model=True)
    elif "VBDDropout" in model_name or "VariationalDropout" in model_name:
        model = LightningTrainingVariationalDropout(gnn_model, model_args, train_args, writer, device, amp = train_args['amp'], init_model=True)
    elif "Drop" in model_name:
        model = LightningTrainingDropout(gnn_model, model_args, train_args, writer, device, amp = train_args['amp'], init_model=True)
    else:
        model = LightningTraining(gnn_model, model_args, train_args, writer, device, amp = train_args['amp'], init_model=True)
    # train model
    if 'pretrained' in train_args and train_args['pretrained']:
        model.load_state_dict(torch.load(train_args['pretrained']))
        print("Load pretrained model from ", train_args['pretrained'])

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(path, "checkpoints"),
        filename="{step}",         # filename is just the step number
        every_n_train_steps=1,     # save after every training step
        save_top_k=0,              # disabled: nothing in the pipeline resumes from
                                   # these, and writing a checkpoint every step is a
                                   # severe bottleneck on network storage (NFS)
        monitor="step",            # monitor the step number
        mode="max",                # larger step number = newer
    )

    trainer = pl.Trainer(accelerator="cuda",
                         max_epochs=train_args['n_epoch'],
                         devices=torch.cuda.device_count(),
                         inference_mode=False,
                         precision="bf16-mixed" if train_args['amp'] else "32",
                         logger = [writer, csv_logger],
                         callbacks=[checkpoint_callback],)

    trainer.fit(model, DataLoader(train_dataset, **train_args['loader_tr_args']), DataLoader(valid_dataset, **train_args['loader_te_args']))
    ### Predict train dataset for pos-hoc calibration
    dp_pred = []
    dp_true = []
    dp_stds = []
    

    if "Evi" in train_args['loss_fn']:
        test_loader = DataLoader(test_dataset, **train_args['loader_te_args'])
        predictions = trainer.predict(model, test_loader)
        kappas = []
        nus = []
        for idx, pred in enumerate(predictions):
            dp_pred_, dp_true_, dp_std_, kappa, nu = pred
            dp_pred.append(dp_pred_.detach().to("cpu"))
            dp_true.append(dp_true_.detach().to("cpu"))
            dp_stds.append(dp_std_.detach().to("cpu"))
            kappas.append(kappa.detach().to("cpu"))
            nus.append(nu.detach().to("cpu"))
    elif "Drop" in model_name:
        test_loader = DataLoader(test_dataset, **train_args['loader_te_args'])
        predictions = trainer.predict(model, test_loader)

        for idx, pred in enumerate(predictions):
            dp_pred_, dp_true_, dp_stds_ = pred
            dp_pred.append(dp_pred_.detach().to("cpu"))
            dp_true.append(dp_true_.detach().to("cpu"))
            dp_stds.append(dp_stds_.detach().to("cpu"))

    else:
        test_loader = DataLoader(test_dataset, **train_args['loader_te_args'])
        predictions = trainer.predict(model, test_loader)
        
        for idx, pred in enumerate(predictions):
            dp_pred_, dp_true_ = pred
            dp_pred.append(dp_pred_.detach().to("cpu"))
            dp_true.append(dp_true_.detach().to("cpu"))

        print("Standard test, generate random energy and force stds")
        dp_stds = [torch.eye(3)[None, :, :].repeat(dp_pred_.size(0), 1, 1) for dp_pred_ in dp_pred]


    # if it is mupltiple gpu, get the current rank
    if trainer.world_size > 1:
        rank = trainer.global_rank
        world_size = trainer.world_size
    else:
        rank = 0
        world_size = 1
    
    # concatenate the results to a single torch tensor
    dp_pred = torch.cat(dp_pred, dim=0)
    dp_true = torch.cat(dp_true, dim=0)
    dp_stds = torch.cat(dp_stds, dim=0)
    Path(path, "train").mkdir(parents=True, exist_ok=True )
    torch.save(dp_pred, Path(path, "train", f"dp_pred_{rank}.pt"))
    torch.save(dp_true, Path(path, "train", f"dp_true_{rank}.pt"))
    torch.save(dp_stds, Path(path, "train", f"dp_stds_{rank}.pt"))

    ### Finish train prediction, start test prediction
    
    
    dp_pred = []
    dp_true = []
    dp_stds = []
    

    if "Evi" in train_args['loss_fn']:
        test_loader = DataLoader(test_dataset, **train_args['loader_te_args'])
        predictions = trainer.predict(model, test_loader)
        kappas = []
        nus = []
        for idx, pred in enumerate(predictions):
            dp_pred_, dp_true_, dp_std_, kappa, nu = pred
            dp_pred.append(dp_pred_.detach().to("cpu"))
            dp_true.append(dp_true_.detach().to("cpu"))
            dp_stds.append(dp_std_.detach().to("cpu"))
            kappas.append(kappa.detach().to("cpu"))
            nus.append(nu.detach().to("cpu"))
    elif "Drop" in model_name:
        test_loader = DataLoader(test_dataset, **train_args['loader_te_args'])
        predictions = trainer.predict(model, test_loader)

        for idx, pred in enumerate(predictions):
            dp_pred_, dp_true_, dp_stds_ = pred
            dp_pred.append(dp_pred_.detach().to("cpu"))
            dp_true.append(dp_true_.detach().to("cpu"))
            dp_stds.append(dp_stds_.detach().to("cpu"))

    else:
        test_loader = DataLoader(test_dataset, **train_args['loader_te_args'])
        predictions = trainer.predict(model, test_loader)
        
        for idx, pred in enumerate(predictions):
            dp_pred_, dp_true_ = pred
            dp_pred.append(dp_pred_.detach().to("cpu"))
            dp_true.append(dp_true_.detach().to("cpu"))

        print("Standard test, generate random energy and force stds")
        dp_stds = [torch.eye(3)[None, :, :].repeat(dp_pred_.size(0), 1, 1) for dp_pred_ in dp_pred]


    # if it is mupltiple gpu, get the current rank
    if trainer.world_size > 1:
        rank = trainer.global_rank
        world_size = trainer.world_size
    else:
        rank = 0
        world_size = 1
    
    # concatenate the results to a single torch tensor
    dp_pred = torch.cat(dp_pred, dim=0)
    dp_true = torch.cat(dp_true, dim=0)
    dp_stds = torch.cat(dp_stds, dim=0)

    torch.save(dp_pred, Path(path, f"dp_pred_{rank}.pt"))
    torch.save(dp_true, Path(path, f"dp_true_{rank}.pt"))
    torch.save(dp_stds, Path(path, f"dp_stds_{rank}.pt"))

    if "Evi" in train_args['loss_fn']:
        kappas = torch.cat(kappas, dim=0)
        nus = torch.cat(nus, dim=0)
        torch.save(kappas, Path(path, f"kappa_{rank}.pt"))
        torch.save(nus, Path(path, f"nu_{rank}.pt"))

    if len(dp_pred.shape) == 3: # it is sampling output
        dp_pred = dp_pred.mean(dim=1)
    dp_mae = float(torch.nn.functional.l1_loss(dp_pred, dp_true))
    print("dipole moment = {:.4f}".format(dp_mae))

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
    if "Evi" in train_args['loss_fn']:
        if "Dropout" in model_args['name']:
            path = Path("./runs" , time.strftime("%m-%d"), train_args['dataset'] +"_FEvi_"+ model_args['name'] 
                + '_type' + str(model_args['dropout_method']) +'_ratio'+"{:.0E}".format(model_args['dropout_p'])
                + '_batch' + str(train_args['loader_tr_args']['batch_size']) +'_lr'
                + "{:.0E}".format(train_args['learning_rate']) + "_" + time.strftime("%H-%M"))
        else:                                                                                                                    
            path = Path("./runs" , time.strftime("%m-%d"), train_args['dataset'] +"_FEvi_"+ model_args['name']
            + '_loss' + str(train_args['loss_fn']) + '_batch' + str(train_args['loader_tr_args']['batch_size']) +'_lr'
            + "{:.0E}".format(train_args['learning_rate']) + "_" + time.strftime("%H-%M"))
    elif ("Dropout" in model_args['name']) and "VBDDropout" not in model_args['name']:
        path = Path("./runs" , time.strftime("%m-%d"), train_args['dataset'] + "_"+ model_args['name']
            + '_type' + str(model_args['dropout_method']) +'_ratio'+"{:.0E}".format(model_args['dropout_p'])+ '_batch' + str(train_args['loader_tr_args']['batch_size']) +'_lr'
            + "{:.0E}".format(train_args['learning_rate']) + "_" + time.strftime("%H-%M"))
    elif "VBDDropout" in model_args['name']:
        path = Path("./runs" , time.strftime("%m-%d"), train_args['dataset'] + "_"+ model_args['name'] 
            + '_kl'+"{:.0E}".format(train_args['kl_weight'])+ '_batch' + str(train_args['loader_tr_args']['batch_size']) +'_lr' 
            + "{:.0E}".format(train_args['learning_rate']) + "_" + time.strftime("%H-%M"))
    else:
        path = Path("./runs" , time.strftime("%m-%d"), train_args['dataset'] + "_"+ model_args['name'] 
            + '_loss' + str(train_args['loss_fn']) +'_kl'+"{:.0E}".format(train_args['kl_weight'])+ '_batch' + str(train_args['loader_tr_args']['batch_size']) +'_lr' 
            + "{:.0E}".format(train_args['learning_rate']) + "_" + time.strftime("%H-%M"))
    path.mkdir(parents=True, exist_ok=True )
    train_args['path'] = path
    train_args['test'] = args.test
    train_args['optimizer_args']['lr'] = train_args['learning_rate']

    


    with open(Path(path, "results.csv"), 'w') as f:
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
                # override per submission with SLURM_EXCLUDE_NODES when a node misbehaves
                slurm_exclude=os.environ.get("SLURM_EXCLUDE_NODES", "p2-gpu-15"), #3-gpu-[43-50]"
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



    
    
