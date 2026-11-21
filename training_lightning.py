#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time: 03/06/23 11:21:25
# File: training.py

import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from ocpmodels.modules.normalizer import Normalizer
from torch.optim.lr_scheduler import StepLR
from torch_ema import ExponentialMovingAverage
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from uncertainty_ai.train.scheduler import LinearWarmupExponentialDecay
from uncertainty_ai.utils import get_loss_fn
from uncertainty_ai.metrics.force_calibration import get_force_cov
import pytorch_lightning as pl
from uncertainty_ai.metrics.force_calibration import get_force_cov, miscalibration_area_vector


class LightningTraining(pl.LightningModule):
    """Function used to train the model

    Args:
        net (torch.nn.Module): The neural network model
        net_args (dict): The arguments for the neural network model
        args (dict): The arguments for training
        writer (SummaryWriter): The tensorboard writer
        device (torch.device): The device to run the model
        amp (bool): Whether to use automatic mixed precision
        init_model (bool): Whether to initialize the model
    """
    def __init__(self, net, net_args, args, writer, device, amp = False, init_model=True):
        super(LightningTraining, self).__init__()
        self.args = args
        self.writer = writer
        self.model = net(**net_args)
        print(self.model)

        # EMP
        if self.args['ema']:
            self.ema = ExponentialMovingAverage(self.model.parameters(), self.args['ema_decay'])
        else:
            self.ema = False

        # Normalizer
        if self.args['normalizer']:
            self.normalizer = Normalizer(**self.args['normalizer_args'])
            self.fnormalizer = Normalizer(mean= 0.0, std = self.args['normalizer_args']['std'])
        else:
            self.normalizer = None
        
        self.save_hyperparameters(ignore=['writer', 'ema', 'normalizer'])
        self.validation_step_outputs = []
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = None
        if self.args['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), **self.args['optimizer_args'])
        elif self.args['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(), **self.args['optimizer_args'])
        elif self.args['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), **self.args['optimizer_args'])
        
        scheduler = None
        if self.args['lr_schedule']:
            print(f"Using learning rate scheduler: {self.args['lr_schedule_args']['name']}")
            if self.args['lr_schedule_args']['name'] == "linear_warmup_exponential_decay":
                lr_schedule_args = copy.deepcopy(self.args['lr_schedule_args'])
                del lr_schedule_args['name']
                scheduler = LinearWarmupExponentialDecay(optimizer, **lr_schedule_args)
            elif self.args['lr_schedule_args']['name'] == "step_lr":
                lr_schedule_args = copy.deepcopy(self.args['lr_schedule_args'])
                del lr_schedule_args['name']
                scheduler = StepLR(optimizer, **lr_schedule_args)
        
        if scheduler:
            return [optimizer], [scheduler]
        return optimizer
    
    def on_train_start(self):
        # Move EMA to the correct device (e.g., GPU)
        device = next(self.parameters()).device  # Get the device of the model
        self.ema.to(device)  # Move EMA to the same device
        print(f"EMA moved to {device}")
    
    def training_step(self, batch, batch_idx):
        loss_fn = get_loss_fn(self.args['loss_fn'], self.args['loss_fn_args'])
        
        f_true = batch.force
        e_true = batch.y
        e_true = self.normalizer.norm(e_true)
        f_true = self.fnormalizer.norm(f_true)
        
        optimizer = self.optimizers()
        optimizer.zero_grad()
        
        e_pred, f_pred, _, _ = self(batch)
        natoms = batch.natoms[batch.batch]
        e_pred = e_pred.view(e_true.size())
        f_pred = f_pred.view(f_true.size())

        loss = loss_fn(e_pred, f_pred, e_true, f_true, natoms)
        temp_energy = torch.zeros([e_true.size(0), 2])
        temp_energy[:, 0] = e_true
        temp_energy[:, 1] = e_pred.view(e_true.size())
        
        # print("energy:", temp_energy)
        self.log('train_loss', loss,  batch_size = e_pred.size(0))
        self.log('lr', optimizer.param_groups[0]['lr'])
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.model.parameters())
    
    def validation_step(self, batch, batch_idx):
        e_pred, f_pred, _, _ = self(batch)
        y = batch.force
        e_true = batch.y
        e_pred = e_pred.view(e_true.size())
        e_pred = self.normalizer.denorm(e_pred)
        f_pred = self.fnormalizer.denorm(f_pred)
        f_pred = f_pred.view(y.size())
        # print("e_pred:", e_pred[:5])
        # print("e_true:", e_true[:5])
        # print("f_pred:", f_pred[:5])
        # print("f_true:", batch.force[:5])
        f_mae = F.l1_loss(f_pred, y)
        e_mae = F.l1_loss(e_pred, e_true)
        # self.validation_step_outputs.append((e_mae, e_true.size(0), f_mae, y.size(0)))
        
        self.log('val_e_mae', e_mae, batch_size = e_pred.size(0), sync_dist=True)
        self.log('val_force_mae', f_mae,  batch_size = e_pred.size(0), sync_dist=True)
        return f_mae * 0.99 + e_mae * 0.01


    # def on_validation_epoch_end(self):
    #     maes, nmols, fmaes, natoms = [], [], [], []
    #     for mae, nmol, fmae, natom in self.validation_step_outputs:
    #         maes.append(mae)
    #         nmols.append(nmol)
    #         fmaes.append(fmae)
    #         natoms.append(natom)
        
    #     maes = torch.stack(maes)
    #     nmols = torch.tensor(nmols, device=maes.device)
    #     fmaes = torch.stack(fmaes)
    #     natoms = torch.tensor(natoms, device=fmaes.device)

    #     avg_mae = (torch.sum(maes *  nmols) / torch.sum(nmols)).item()
    #     avg_fmae = (torch.sum(fmaes * natoms) / torch.sum(natoms)).item()

    #     self.log('val_e_mae', avg_mae,  sync_dist=True)
    #     self.log('val_force_mae', avg_fmae,  sync_dist=True)
    #     self.validation_step_outputs.clear()  # free memory
    
    def predict_step(self, batch, batch_idx):
        e_pred, f_pred, _, _ = self(batch)
        y = batch.force
        e_true = batch.y
        e_pred = e_pred.view(e_true.size())
        e_pred = self.normalizer.denorm(e_pred)
        f_pred = f_pred.view(y.size())
        f_pred = self.fnormalizer.denorm(f_pred)
        
        # Calculate uncertainty from evidential parameters
        # For force uncertainty, we can use the diagonal of the covariance matrix
        # Psi is the precision matrix, so uncertainty = 1/sqrt(diag(Psi))
        # force_uncertainty = torch.eye(3, device=f_pred.device).view(1, 3, 3)
        # # repeat it to be B, 3, 3
        # force_uncertainty = force_uncertainty.repeat(f_pred.size(0), 1, 1)
        
        return e_pred, e_true, f_pred, y
    

class LightningTrainingEvidential(LightningTraining):
    """Function used to train the model

    Args:
        net (torch.nn.Module): The neural network model
        net_args (dict): The arguments for the neural network model
        args (dict): The arguments for training
        writer (SummaryWriter): The tensorboard writer
        device (torch.device): The device to run the model
        amp (bool): Whether to use automatic mixed precision
        init_model (bool): Whether to initialize the model
    """
    def __init__(self, net, net_args, args, writer, device, amp = False, init_model=True):
        super(LightningTrainingEvidential, self).__init__(net, net_args, args, writer, device, amp, init_model)
    
    def training_step(self, batch, batch_idx):
        loss_fn = get_loss_fn(self.args['loss_fn'], self.args['loss_fn_args'])
        loss_fn = loss_fn.to(self.device)
        y = batch.force
        y = self.fnormalizer.norm(y)
        e_true = batch.y
        e_true = self.normalizer.norm(e_true)

        optimizer = self.optimizers()
        optimizer.zero_grad()

        e_pred, f_pred = self.forward(batch)

        natoms = batch.natoms[batch.batch]
        min_val = 1e-6
        pred, loglambdas, logalphas, logbetas = torch.split(e_pred,
                                                        e_pred.shape[1]//4,
                                                        dim=1)
        lambdas = F.softplus(loglambdas) + min_val
        alphas = F.softplus(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
        betas = F.softplus(logbetas) + min_val

        pred = pred.view(e_true.size())
        f_pred = f_pred.view(y.size())

        loss = loss_fn(pred, lambdas, alphas, betas, f_pred, e_true, y, natoms)

        return loss


    def validation_step(self, batch, batch_idx):
        energy, f_pred = self.forward(batch)
        min_val = 1e-6

        e_pred, _, _, _ = torch.split(energy,
                                    energy.shape[1]//4,
                                    dim=1)
        y = batch.force
        e_true = batch.y
        e_pred = e_pred.view(e_true.size())
        e_pred = self.normalizer.denorm(e_pred)
        f_pred = f_pred.view(y.size())
        f_pred = self.fnormalizer.denorm(f_pred)
        
        f_mae = F.l1_loss(f_pred, y)
        e_mae = F.l1_loss(e_pred, e_true)
        self.validation_step_outputs.append((e_mae, e_true.size(0), f_mae, y.size(0)))
        
        return f_mae * 0.99 + e_mae * 0.01

    
    def predict_step(self, batch, batch_idx):
        energy, f_pred = self.forward(batch)
        min_val = 1e-6

        e_pred, loglambdas, logalphas, logbetas = torch.split(energy,
                                                                energy.shape[1]//4,
                                                                dim=1)
        lambdas = F.softplus(loglambdas) + min_val
        alphas = F.softplus(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
        betas = F.softplus(logbetas) + min_val
        e_std = betas/(lambdas*(alphas - 1))**0.5
        y = batch.force
        e_true = batch.y
        e_pred = e_pred.view(e_true.size())
        e_pred = self.normalizer.denorm(e_pred)
        f_pred = f_pred.view(y.size())
        f_pred = self.fnormalizer.denorm(f_pred)
        return e_pred, e_true, f_pred, y, e_std
    
class LightningTrainingForceEvidential(LightningTraining):
    """Function used to train the model

    Args:
        net (torch.nn.Module): The neural network model
        net_args (dict): The arguments for the neural network model
        args (dict): The arguments for training
        writer (SummaryWriter): The tensorboard writer
        device (torch.device): The device to run the model
        amp (bool): Whether to use automatic mixed precision
        init_model (bool): Whether to initialize the model
    """
    def __init__(self, net, net_args, args, writer, device, amp = False, init_model=True):
        super(LightningTrainingForceEvidential, self).__init__(net, net_args, args, writer, device, amp, init_model)
        self.loss_fn = get_loss_fn(self.args['loss_fn'], self.args['loss_fn_args'])
    
    def training_step(self, batch, batch_idx):
        loss_fn = self.loss_fn.to(self.device)
        f_true = batch.force
        #! y = self.fnormalizer.norm(y) 
        e_true = batch.y
        e_true = self.normalizer.norm(e_true) # Note we did not normalize force, so the energy std must be 1 to avoid scale issue

        optimizer = self.optimizers()
        optimizer.zero_grad()

        e_pred, f_pred, Psi, nu = self.forward(batch)

        natoms = batch.natoms[batch.batch]
        f_pred = f_pred.view(f_true.size())
        e_pred = e_pred.view(e_true.size())
        loss, kappa, nu = loss_fn(e_pred, f_pred, Psi, nu, e_true, f_true, natoms)
        self.log("train_loss", loss, batch_size = e_pred.size(0))
        self.log("kappa", kappa.mean(), batch_size = e_pred.size(0))
        self.log("nu", nu.mean(), batch_size = e_pred.size(0))
        return loss


    def validation_step(self, batch, batch_idx):
        e_pred, f_pred, Psi, nu = self.forward(batch)
        
        f_true = batch.force
        e_true = batch.y
        f_pred = f_pred.view(batch.force.size())
        e_pred = e_pred.view(e_true.size())

        e_pred = self.normalizer.denorm(e_pred)
        alea, epis, _, _ = self.loss_fn.get_uncertainties(f_pred, Psi, nu)

        f_mae = F.l1_loss(f_pred, f_true)
        e_mae = F.l1_loss(e_pred, e_true)
        miscal_area = miscalibration_area_vector(f_pred, alea+ epis, f_true)
        
        self.log('val_fmiscalibration_area', miscal_area, batch_size = f_true.size(0), sync_dist=True)
        self.log('val_f_mae', f_mae, batch_size = f_true.size(0), sync_dist=True)
        self.log('val_e_mae', e_mae, batch_size = e_pred.size(0), sync_dist=True)
        return f_mae * 0.99 + e_mae * 0.01

    
    def predict_step(self, batch, batch_idx):
        e_pred, f_pred, Psi, nu = self.forward(batch)
        
        f_true = batch.force
        e_true = batch.y
        f_pred = f_pred.view(batch.force.size())
        e_pred = e_pred.view(e_true.size())

        e_pred = self.normalizer.denorm(e_pred)

        alea, epis, kappa, nu = self.loss_fn.get_uncertainties(f_pred, Psi, nu)
        return e_pred, e_true, f_pred, f_true, alea+epis, kappa, nu
    
    # def on_predict_epoch_end(self, outputs):
    #     print("before gahtering", len(outputs))
    #     # Gather data across all processes
        
    #     if self.trainer.world_size > 1:
    #         outputs = self.all_gather(outputs)
    #         print("after gathering", len(outputs))
        
    #     return outputs
        #     e_pred = self.trainer.strategy.all_gather(e_pred)
        #     e_true = self.trainer.strategy.all_gather(e_true)
        #     f_pred = self.trainer.strategy.all_gather(f_pred)
        #     y = self.trainer.strategy.all_gather(y)
        #     e_std = self.trainer.strategy.all_gather(e_std)
        # e_pred = self.trainer.strategy.all_gather(e_pred)
        # e_true = self.trainer.strategy.all_gather(e_true)
        # f_pred = self.trainer.strategy.all_gather(f_pred)
        # y = self.trainer.strategy.all_gather(y)
        # e_std = self.trainer.strategy.all_gather(e_std)

class LightningTrainingForceEvidentialTwoStage(LightningTraining):
    """Two-stage training for force evidential models"""
    def __init__(self, net, net_args, args, writer, device, amp=False, init_model=True):
        super().__init__(net, net_args, args, writer, device, amp, init_model)
        # Stage 1: Use standard loss (EnergyL1ForceL2Loss)
        self.stage1_loss_fn = get_loss_fn("EnergyL1ForceL2Loss", self.args.get('stage1_loss_args', {}))
        # Stage 2: Use evidential loss (EnergyL1ForceEvidentialLoss)
        self.stage2_loss_fn = get_loss_fn("EnergyL1ForceEvidentialLoss", self.args.get('stage2_loss_args', {}))
        self.stage = 1  # 1: train all, 2: train only cov_decoder
        self.stage_switch_epoch = self.args.get('stage_switch_epoch', 50)
        
        # Separate learning rate schedulers for each stage
        self.stage1_scheduler = None
        self.stage2_scheduler = None
        self._setup_stage_schedulers()
        
        print("Starting training in stage 1")
        print(f"Stage 1 loss: EnergyL1ForceL2Loss")
        print(f"Stage 2 loss: EnergyL1ForceEvidentialLoss")
    
    def _setup_stage_schedulers(self):
        """Setup separate learning rate schedulers for each stage"""
        # Store scheduler configurations for later creation
        self.stage1_scheduler_config = None
        self.stage2_scheduler_config = None
        
        # Stage 1 scheduler configuration
        if self.args.get('stage1_lr_schedule', False):
            stage1_lr_args = self.args.get('stage1_lr_schedule_args', {})
            if stage1_lr_args.get('name') == "linear_warmup_exponential_decay":
                lr_args = copy.deepcopy(stage1_lr_args)
                del lr_args['name']
                self.stage1_scheduler_config = lr_args
        
        # Stage 2 scheduler configuration
        if self.args.get('stage2_lr_schedule', False):
            stage2_lr_args = self.args.get('stage2_lr_schedule_args', {})
            if stage2_lr_args.get('name') == "linear_warmup_exponential_decay":
                lr_args = copy.deepcopy(stage2_lr_args)
                del lr_args['name']
                self.stage2_scheduler_config = lr_args
    
    def on_train_start(self):
        super().on_train_start()
        print(f"Starting training in stage {self.stage}")
        print(f"Stage 1 loss: EnergyL1ForceL2Loss")
        print(f"Stage 2 loss: EnergyL1ForceEvidentialLoss")
        
    def on_train_epoch_start(self):
        """Check if we need to switch stages"""
        current_epoch = self.current_epoch
        if current_epoch == self.stage_switch_epoch and self.stage == 1:
            self.switch_to_stage_2()
            
    def switch_to_stage_2(self):
        """Switch to stage 2: freeze backbone, train decoder and cov_decoder"""
        print(f"Switching to stage 2 at epoch {self.current_epoch}")
        self.stage = 2
        self.model.freeze_backbone()
        
        # Create new optimizer for stage 2 (cov_decoder and decoder parameters)
        params = list(self.model.cov_decoder.parameters()) + list(self.model.decoder.parameters())
        if self.args['optimizer'] == 'Adam':
            optimizer = optim.Adam(params,  **self.args['optimizer_args'])
        elif self.args['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(params,  **self.args['optimizer_args'])
        elif self.args['optimizer'] == 'SGD':
            optimizer = optim.SGD(params,  **self.args['optimizer_args'])
        
        print(f"Stage 2 optimizer created with {sum(p.numel() for p in optimizer.param_groups[0]['params'])} parameters")
        print(f"Stage 2 fixed learning rate: {optimizer.param_groups[0]['lr']}")
        
        # Update trainer optimizers (no scheduler for stage 2)
        self.trainer.optimizers = [optimizer]
        self.trainer.lr_schedulers = []  # No scheduler for stage 2
        
        print("Backbone frozen, decoder and cov_decoder will be trained")
        print("Switched to EnergyL1ForceEvidentialLoss with fixed LR=5e-4")
        
    def configure_optimizers(self):
        """Configure optimizers with stage-specific behavior"""
        if self.stage == 1:
            # Stage 1: Optimize all parameters with scheduler
            if self.args['optimizer'] == 'Adam':
                optimizer = optim.Adam(self.model.parameters(), **self.args['optimizer_args'])
            elif self.args['optimizer'] == 'AdamW':
                optimizer = optim.AdamW(self.model.parameters(), **self.args['optimizer_args'])
            elif self.args['optimizer'] == 'SGD':
                optimizer = optim.SGD(self.model.parameters(), **self.args['optimizer_args'])
            
            # For stage 1, use the stage1_scheduler if configured
            if hasattr(self, 'stage1_scheduler_config') and self.stage1_scheduler_config is not None:
                self.stage1_scheduler = LinearWarmupExponentialDecay(optimizer, **self.stage1_scheduler_config)
                return [optimizer], [self.stage1_scheduler]
            # Fallback to original scheduler logic
            else:
                scheduler = None
                if self.args['lr_schedule']:
                    print(f"Using learning rate scheduler: {self.args['lr_schedule_args']['name']}")
                    if self.args['lr_schedule_args']['name'] == "linear_warmup_exponential_decay":
                        lr_schedule_args = copy.deepcopy(self.args['lr_schedule_args'])
                        del lr_schedule_args['name']
                        scheduler = LinearWarmupExponentialDecay(optimizer, **lr_schedule_args)
                    elif self.args['lr_schedule_args']['name'] == "step_lr":
                        lr_schedule_args = copy.deepcopy(self.args['lr_schedule_args'])
                        del lr_schedule_args['name']
                        scheduler = StepLR(optimizer, **lr_schedule_args)
                
                if scheduler:
                    return [optimizer], [scheduler]
                return optimizer
        else:
            # Stage 2: This should not be called since we handle stage 2 in switch_to_stage_2
            # But provide a fallback just in case
            if self.args['optimizer'] == 'Adam':
                optimizer = optim.Adam(self.model.cov_decoder.parameters(), lr=5e-4, **self.args['optimizer_args'])
            elif self.args['optimizer'] == 'AdamW':
                optimizer = optim.AdamW(self.model.cov_decoder.parameters(), lr=5e-4, **self.args['optimizer_args'])
            elif self.args['optimizer'] == 'SGD':
                optimizer = optim.SGD(self.model.cov_decoder.parameters(), lr=5e-4, **self.args['optimizer_args'])
            return optimizer
    
    def training_step(self, batch, batch_idx):
        """Training step with different loss functions for each stage"""
        optimizer = self.optimizers()
        optimizer.zero_grad()

        if self.stage == 1:
            # Stage 1: Use standard EnergyL1ForceL2Loss
            loss_fn = self.stage1_loss_fn.to(self.device)
            f_true = batch.force
            e_true = batch.y
            e_true = self.normalizer.norm(e_true)
            f_true = self.fnormalizer.norm(f_true)
            
            # Model always returns 4 values, but we only need e_pred and f_pred for stage 1
            e_pred, f_pred, Psi, nu = self.forward(batch)
            natoms = batch.natoms[batch.batch]
            e_pred = e_pred.view(e_true.size())
            f_pred = f_pred.view(f_true.size())
            
            loss = loss_fn(e_pred, f_pred, e_true, f_true, natoms)
            self.log('train_loss', loss, batch_size = e_pred.size(0))
            self.log('lr', optimizer.param_groups[0]['lr'])
            self.log("stage", self.stage, batch_size=e_pred.size(0))
            
            # Debug: Log scheduler info for stage 1
            if hasattr(self, 'stage1_scheduler') and self.stage1_scheduler is not None:
                self.log('stage1_scheduler_step', self.stage1_scheduler._step_count, batch_size=e_pred.size(0))
            
            return loss
            
        else:
            # Stage 2: Use evidential EnergyL1ForceEvidentialLoss
            loss_fn = self.stage2_loss_fn.to(self.device)
            f_true = batch.force
            e_true = batch.y
            e_true = self.normalizer.norm(e_true)
            
            e_pred, f_pred, Psi, nu = self.forward(batch)
            natoms = batch.natoms[batch.batch]
            f_pred = f_pred.view(f_true.size())
            e_pred = e_pred.view(e_true.size())
            
            loss, kappa, nu_val = loss_fn(e_pred, f_pred, Psi, nu, e_true, f_true, natoms)
            self.log("train_loss", loss, batch_size = e_pred.size(0))
            self.log("kappa", kappa.mean(), batch_size = e_pred.size(0))
            self.log("nu", nu_val.mean(), batch_size = e_pred.size(0))
            self.log('lr', optimizer.param_groups[0]['lr'])
            self.log("stage", self.stage, batch_size=e_pred.size(0))
            
            return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - handle both stages"""
        if self.stage == 1:
            # Stage 1: Standard validation
            e_pred, f_pred, Psi, nu = self.forward(batch)
            y = batch.force
            e_true = batch.y
            e_pred = e_pred.view(e_true.size())
            e_pred = self.normalizer.denorm(e_pred)
            f_pred = f_pred.view(y.size())
            
            f_mae = F.l1_loss(f_pred, y)
            e_mae = F.l1_loss(e_pred, e_true)
            
            self.log('val_e_mae', e_mae, batch_size=e_pred.size(0), sync_dist=True)
            self.log('val_f_mae', f_mae, batch_size=e_pred.size(0), sync_dist=True)
            
        else:
            # Stage 2: Evidential validation
            e_pred, f_pred, Psi, nu = self.forward(batch)
            
            f_true = batch.force
            e_true = batch.y
            f_pred = f_pred.view(batch.force.size())
            e_pred = e_pred.view(e_true.size())

            e_pred = self.normalizer.denorm(e_pred)
            alea, epis, _, _ = self.stage2_loss_fn.get_uncertainties(f_pred, Psi, nu)

            f_mae = F.l1_loss(f_pred, f_true)
            e_mae = F.l1_loss(e_pred, e_true)
            miscal_area = miscalibration_area_vector(f_pred, alea + epis, f_true)
            
            self.log('val_fmiscalibration_area', miscal_area, batch_size=f_true.size(0), sync_dist=True)
            self.log('val_f_mae', f_mae, batch_size=f_true.size(0), sync_dist=True)
            self.log('val_e_mae', e_mae, batch_size=e_pred.size(0), sync_dist=True)
        
        return f_mae * 0.99 + e_mae * 0.01

    def predict_step(self, batch, batch_idx):
        """Prediction step - handle both stages"""
        if self.stage == 1:
            # Stage 1: Standard prediction
            e_pred, f_pred, Psi, nu = self.forward(batch)
            y = batch.force
            e_true = batch.y
            e_pred = e_pred.view(e_true.size())
            e_pred = self.normalizer.denorm(e_pred)
            f_pred = f_pred.view(y.size())
            f_pred = self.fnormalizer.denorm(f_pred)
            return e_pred, e_true, f_pred, y
        else:
            # Stage 2: Evidential prediction
            e_pred, f_pred, Psi, nu = self.forward(batch)
            
            f_true = batch.force
            e_true = batch.y
            f_pred = f_pred.view(batch.force.size())
            e_pred = e_pred.view(e_true.size())

            e_pred = self.normalizer.denorm(e_pred)
            alea, epis, kappa, nu = self.stage2_loss_fn.get_uncertainties(f_pred, Psi, nu)
            return e_pred, e_true, f_pred, f_true, alea + epis, kappa, nu

class LightningTrainingDropout(LightningTraining):
    """Function used to train the model

    Args:
        net (torch.nn.Module): The neural network model
        net_args (dict): The arguments for the neural network model
        args (dict): The arguments for training
        writer (SummaryWriter): The tensorboard writer
        device (torch.device): The device to run the model
        amp (bool): Whether to use automatic mixed precision
        init_model (bool): Whether to initialize the model
    """
    def __init__(self, net, net_args, args, writer, device, amp = False, init_model=True):
        super(LightningTrainingDropout, self).__init__(net, net_args, args, writer, device, amp, init_model)
    
    def validation_step(self, batch, batch_idx):
        self.train()
        e_pred, f_pred, _, _ = self(batch)
        y = batch.force
        e_true = batch.y
        e_pred = e_pred.view(e_true.size())
        e_pred = self.normalizer.denorm(e_pred)
        f_pred = self.fnormalizer.denorm(f_pred)
        f_pred = f_pred.view(y.size())
        
        f_mae = F.l1_loss(f_pred, y)
        e_mae = F.l1_loss(e_pred, e_true)
        # self.validation_step_outputs.append((e_mae, e_true.size(0), f_mae, y.size(0)))
        
        self.log('val_e_mae', e_mae, batch_size = e_pred.size(0), sync_dist=True)
        self.log('val_f_mae', f_mae,  batch_size = e_pred.size(0), sync_dist=True)
        return f_mae * 0.99 + e_mae * 0.01
    
    def predict_step(self, batch, batch_idx):
        batch = batch.to(self.device)
        n_drop = 5
        f_true = batch.force
        e_true = batch.y
        
        e_preds = torch.zeros([e_true.size(0), n_drop, 1])
        f_preds = torch.zeros([f_true.size(0), n_drop, 3])
        e_preds = e_preds.to(self.device)
        f_preds = f_preds.to(self.device)

        self.train()
        for i in range(n_drop):
            temp_epred = []
            temp_fpred = []
            # print('n_drop {}/{}'.format(i + 1, n_drop))

            if self.ema:
                with self.ema.average_parameters():
                    energy, force, _, _ = self.model(batch)
            else:
                energy, force, _, _ = self.model(batch)

            energy = self.normalizer.denorm(energy)
            force = self.fnormalizer.denorm(force)

            e_preds[:, i, :] = energy.view(e_preds[:, i, :].size())
            f_preds[:, i, :] = force
            
            # print("The MAE of {}th drop is {}".format(i, F.l1_loss(e_preds[:, i, :], e_true)))
        
        e_pred  = torch.mean(e_preds, dim = 1)
        f_pred = torch.mean(f_preds, dim = 1)
        e_std = torch.std(e_preds, dim = 1)
        f_cov = get_force_cov(f_preds)

        e_pred = e_pred.view(e_true.size())
        f_pred = f_pred.view(f_true.size())
        e_std = e_std.view(e_true.size())

        return e_pred, e_true, f_pred, f_true, e_std, f_cov
    
class LightningTrainingVariationalDropout(LightningTrainingDropout):
    """Function used to train the model

    Args:
        net (torch.nn.Module): The neural network model
        net_args (dict): The arguments for the neural network model
        args (dict): The arguments for training
        writer (SummaryWriter): The tensorboard writer
        device (torch.device): The device to run the model
        amp (bool): Whether to use automatic mixed precision
        init_model (bool): Whether to initialize the model
    """
    def __init__(self, net, net_args, args, writer, device, amp = False, init_model=True):
        super(LightningTrainingVariationalDropout, self).__init__(net, net_args, args, writer, device, amp, init_model)
    
    def training_step(self, batch, batch_idx):
        loss_fn = get_loss_fn(self.args['loss_fn'], self.args['loss_fn_args'])
        
        f_true = batch.force
        e_true = batch.y
        e_true = self.normalizer.norm(e_true)
        f_true = self.fnormalizer.norm(f_true)
        
        optimizer = self.optimizers()
        optimizer.zero_grad()
        
        e_pred, f_pred = self(batch)
        natoms = batch.natoms[batch.batch]
        e_pred = e_pred.view(e_true.size())
        f_pred = f_pred.view(f_true.size())
        loss = loss_fn(e_pred, f_pred, e_true, f_true, natoms)
        
        kl = self.model.kl() * self.args['kl_weight']
        loss += kl
        self.log('kl', kl,  batch_size = e_pred.size(0))
        
        self.log('train_loss', loss,  batch_size = e_pred.size(0))
        return loss
    



