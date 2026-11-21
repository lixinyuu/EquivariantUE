#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: xinyu li
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
from uncertainty_ai.metrics.force_calibration import get_force_cov, miscalibration_area_vector
import pytorch_lightning as pl


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
        
        dp_true = batch.y
        optimizer = self.optimizers()
        optimizer.zero_grad()
        dp_pred, _, _ = self(batch)
        dp_pred = dp_pred.view(dp_true.size())

        loss = loss_fn(dp_pred, dp_true)
        
        # print("energy:", temp_energy)
        self.log('train_loss', loss,  batch_size = batch.y.size(0))
        self.log('lr', optimizer.param_groups[0]['lr'])
        return loss

    def optimizer_step(self, *args, **kwargs):
        # Zero out NaN/inf gradients before the optimizer step to prevent
        # weight corruption when the evidential loss produces degenerate gradients.
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.model.parameters())
    
    def validation_step(self, batch, batch_idx):
        dp_true = batch.y
        dp_pred, _, _ = self(batch)
        dp_pred = dp_pred.view(dp_true.size())
        dp_mae = F.l1_loss(dp_true, dp_pred)
        
        self.log('val_dp_mae', dp_mae, batch_size = dp_true.size(0), sync_dist=True)
        return dp_mae


    
    def predict_step(self, batch, batch_idx):
        dp_true = batch.y
        dp_pred, _, _ = self(batch)
        dp_pred = dp_pred.view(dp_true.size())
        return dp_pred, dp_true
    

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
        dp_true = batch.y
        dp_pred, _, _ = self(batch)
        dp_pred = dp_pred.view(dp_true.size())
        dp_mae = F.l1_loss(dp_true, dp_pred)
        
        self.log('val_dp_mae', dp_mae, batch_size = dp_true.size(0), sync_dist=True)
        return dp_mae

    
    def predict_step(self, batch, batch_idx):
        batch = batch.to(self.device)
        n_drop = 5

        dp_true = batch.y
        dp_preds = torch.zeros([dp_true.size(0), n_drop, 3])
        
        self.train()
        for i in range(n_drop):

            if self.ema:
                with self.ema.average_parameters():
                    dp_pred, _, _ = self(batch)
                    dp_pred = dp_pred.view(dp_true.size())
            else:
                dp_pred, _, _ = self(batch)
                dp_pred = dp_pred.view(dp_true.size())


            dp_preds[:, i, :] = dp_pred
            
            # print("The MAE of {}th drop is {}".format(i, F.l1_loss(e_preds[:, i, :], e_true)))
        
        dp_pred = torch.mean(dp_preds, dim = 1)
        dp_cov = get_force_cov(dp_preds)
        
        return dp_preds, dp_true, dp_cov




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
        dp = batch.y
        

        optimizer = self.optimizers()
        optimizer.zero_grad()

        dp_pred, Psi, lognu = self.forward(batch)
        min_val = 1e-6
        dp_pred = dp_pred.squeeze(-1)

        loss, kappa, nu = loss_fn(dp_pred, Psi, lognu, dp)
        
        self.log('train_loss', loss,  batch_size = batch.y.size(0))
        self.log('lr', optimizer.param_groups[0]['lr'])
        self.log('nu', nu.mean(), batch_size = batch.y.size(0))
        return loss


    def validation_step(self, batch, batch_idx):
        dp_pred, Psi, lognu = self.forward(batch)

        dp_pred = dp_pred.squeeze(-1)                 # (B,3)
        dp_true = batch.y                              # (B,3)
        dp_mae = F.l1_loss(dp_true, dp_pred)
        self.log('val_dp_mae', dp_mae, batch_size = dp_true.size(0), sync_dist=True)

        if torch.isfinite(dp_pred).all() and torch.isfinite(Psi).all() and torch.isfinite(lognu).all():
            alea, epis, _, _ = self.loss_fn.get_uncertainties(dp_pred, Psi, lognu)
            cov = alea + epis
            _, chol_info = torch.linalg.cholesky_ex(cov.to(torch.float64))
            if torch.isfinite(cov).all() and (chol_info == 0).all():
                miscal_area = miscalibration_area_vector(dp_pred, cov, dp_true)
                self.log('val_miscalibration_area', miscal_area, batch_size = dp_true.size(0), sync_dist=True)
            # Paper-time behaviour: a non-PD/non-finite validation covariance only skips this
            # metric. The raise added in 20ed8a0 aborted seeds the published runs kept training.
        return dp_mae

    
    def predict_step(self, batch, batch_idx):
        dp_pred, Psi, lognu = self.forward(batch)
        dp_true = batch.y                              # (B,3)
        dp_pred = dp_pred.view(dp_true.size())
        alea, epis, kappa, nu = self.loss_fn.get_uncertainties(dp_pred, Psi, lognu)
        return dp_pred, dp_true, alea + epis, kappa, nu
    



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
        
        dp_true = batch.y
        optimizer = self.optimizers()
        optimizer.zero_grad()
        dp_pred, _, _ = self(batch)
        dp_pred = dp_pred.view(dp_true.size())

        loss = loss_fn(dp_pred, dp_true)
        
        kl = self.model.kl() * self.args['kl_weight']
        loss += kl
        self.log('kl', kl,  batch_size = dp_pred.size(0))
        
        self.log('train_loss', loss,  batch_size = dp_pred.size(0))
        return loss
    