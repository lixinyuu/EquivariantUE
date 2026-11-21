#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time: 03/08/23 19:20:00
# File: utils.py

import numpy as np
import torch
# from ocpmodels.models.schnet import SchNetWrap

def get_loss_fn(name, loss_args = None):
    if name == "MSELoss":
        from torch.nn import MSELoss
        return MSELoss()
    elif name == 'ForceEvidentialLoss':
        from uncertainty_ai.loss.force_loss import ForceEvidentialLoss
        return ForceEvidentialLoss(**loss_args)
    elif name == "EnergyL1ForceL2Loss":
        from uncertainty_ai.loss.loss import EnergyL1ForceL2Loss
        return EnergyL1ForceL2Loss(**loss_args)
    elif name == "EnergyL2ForceL2Loss":
        from uncertainty_ai.loss.loss import EnergyL2ForceL2Loss
        return EnergyL2ForceL2Loss(**loss_args)
    elif name == "EnergyEvidentialForceL2Loss":
        from uncertainty_ai.loss.loss import EnergyEvidentialForceL2Loss
        return EnergyEvidentialForceL2Loss(**loss_args)
    elif name == "EnergyL1ForceEvidentialLoss":
        from uncertainty_ai.loss.loss import EnergyL1ForceEvidentialLoss
        return EnergyL1ForceEvidentialLoss(**loss_args)
    elif name == "L1Loss":
        from torch.nn import L1Loss
        return L1Loss()
    else:
        raise NotImplementedError()

def test_get_loss_fn():
    loss_fn = get_loss_fn("MSELoss")
    assert isinstance(loss_fn, torch.nn.modules.loss.MSELoss)
    loss_fn = get_loss_fn("RegressionEvidentialLoss", loss_args = {"alpha": 0.1})
    assert isinstance(loss_fn, uncertainty_ai.loss.evidential_loss.RegressionEvidentialLoss)
    loss_fn = get_loss_fn("UncertaintyMSELoss")
    assert isinstance(loss_fn, uncertainty_ai.loss.uncertainty_loss.UncertaintyMSELoss)   



def get_model_args(model, args):
    if model == "PaiNN":
        from uncertainty_ai.models.painn import PaiNN

        default_args = {"num_atoms":1,"bond_feat_dim":1,"num_targets":1,
                "n_atom_basis":128, "n_interactions":3, "n_rbf":20, "cutoff":5.0,
                "activation":"silu", "epsilon":1e-8, "regress_forces": True, "readout":"sum",
                "use_pbc": True, "otf_graph": False, "use_alpha": True, "l_dim": 1, "return_latent": False}
        for k, v in args.items():
            default_args[k] = v
        return PaiNN, default_args
    elif model == "PaiNNMER":
        from uncertainty_ai.models.painn import PaiNNMER

        default_args = {"num_atoms":1,"bond_feat_dim":1,"num_targets":1,
                "n_atom_basis":128, "n_interactions":3, "n_rbf":20, "cutoff":5.0,
                "activation":"silu", "epsilon":1e-8, "regress_forces": True, "readout":"sum",
                "use_pbc": True, "otf_graph": False}
        for k, v in args.items():
            default_args[k] = v
        return PaiNNMER, default_args
    elif model in ["PaiNNDropout22", "PaiNNDropout23"]:
        from uncertainty_ai.models.painn.painndropout import PaiNNDropout22, PaiNNDropout23

        default_args = {"num_atoms":1,"bond_feat_dim":1,"num_targets":1,
                "n_atom_basis":128, "n_interactions":3, "n_rbf":20, "cutoff":5.0,
                "activation":"silu", "epsilon":1e-8, "regress_forces": True, "readout":"sum",
                "use_pbc": True, "otf_graph": False, "dropout_method": "bernoulli", "dropout_p": 0.1, "use_alpha": True, "l_dim": 1}
        for k, v in args.items():
            default_args[k] = v
        if model == "PaiNNDropout22":
            return PaiNNDropout22, default_args
        elif model == "PaiNNDropout23":
            return PaiNNDropout23, default_args
    
    elif model == "PaiNNDropout22MER":
        from uncertainty_ai.models.painn.painndropout import PaiNNDropout22MER

        default_args = {"num_atoms":1,"bond_feat_dim":1,"num_targets":1,
                "n_atom_basis":128, "n_interactions":3, "n_rbf":20, "cutoff":5.0,
                "activation":"silu", "epsilon":1e-8, "regress_forces": True, "readout":"sum",
                "use_pbc": True, "otf_graph": False, "dropout_method": "bernoulli", "dropout_p": 0.1}
        for k, v in args.items():
            default_args[k] = v
        return PaiNNDropout22MER, default_args
    elif model == "MACE2":
        from uncertainty_ai.models.mace2.mace import MACE
        default_args = {
            "num_atoms":1,
            "bond_feat_dim":1,  # not used
            "num_targets":1,
            "r_max":5.0,
            "num_bessel":8,
            "num_polynomial_cutoff":5,
            "max_ell":3,
            "num_interactions":2,
            "hidden_irreps":"128x0e+128x1o",
            "MLP_irreps":"16x0e",
            "correlation": 3,
            "regress_forces": True,  # whether to regress forces
            "use_pbc": False,  # whether to use periodic boundary conditions
            "otf_graph": False,  # whether to use on-the-fly graph generation
        }
        for k, v in args.items():
            default_args[k] = v
        return MACE, default_args
    elif model == "MACEDropout22":
        from uncertainty_ai.models.mace2.mace import MACEDropout22
        default_args = {
            "num_atoms":1,
            "bond_feat_dim":1,  # not used
            "num_targets":1,
            "r_max":5.0,
            "num_bessel":8,
            "num_polynomial_cutoff":5,
            "max_ell":3,
            "num_interactions":2,
            "hidden_irreps":"128x0e+128x1o",
            "MLP_irreps":"16x0e",
            "correlation": 3,
            "regress_forces": True,  # whether to regress forces
            "use_pbc": False,  # whether to use periodic boundary conditions
            "otf_graph": False,  # whether to use on-the-fly graph generation
            "dropout_method": "bernoulli",
            "dropout_p": 0.1,}
        for k, v in args.items():
            default_args[k] = v
        return MACEDropout22, default_args
    else:
        raise NotImplementedError()

    
def has_tensor_cores():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")
    
    # Get the current device index (default is the first GPU)
    device_index = torch.cuda.current_device()
    
    # Get the compute capability of the GPU
    capability = torch.cuda.get_device_capability(device_index)
    
    # Tensor Cores are available in compute capability 7.0 and higher
    # Tensor Cores are in Volta (7.0), Turing (7.5), and Ampere (8.0+) architectures
    major, minor = capability
    if major >= 7:
        return True
    return False


def test_get_loss_fn():
    loss_fn = get_loss_fn('MSELoss')
    print(loss_fn)
    loss_fn = get_loss_fn('RegressionEvidentialLoss', loss_args = {'lam': 0.2})
    print(loss_fn)  


