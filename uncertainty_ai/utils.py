#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: xinyu li
# Time: 03/08/23 19:20:00
# File: utils.py

import numpy as np
import torch


from uncertainty_ai.models.painn.painnvbddropout import PaiNNVBDDropout2, PaiNNVBDDropout22
def get_loss_fn(name, loss_args = None):
    if name == "MSELoss":
        from torch.nn import MSELoss
        return MSELoss()
    elif name  == "L1Loss":
        from torch.nn import L1Loss
        return L1Loss()
    elif name == 'AtomwiseL2Loss':
        from uncertainty_ai.loss.force_loss import AtomwiseL2Loss
        return AtomwiseL2Loss()
    elif name == 'EnsembleAtomwiseL2Loss':
        from uncertainty_ai.loss.ensemble_loss import EnsembleAtomwiseL2Loss
        return EnsembleAtomwiseL2Loss()
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
    elif name == "L1Loss":
        from torch.nn import L1Loss
        return L1Loss()
    elif name == 'RegressionEvidentialLoss':
        from uncertainty_ai.loss.evidential_loss import RegressionEvidentialLoss
        return RegressionEvidentialLoss(**loss_args)
    elif name == "VectorRegressionEvidentialLoss":
        from uncertainty_ai.loss.vector_evidential_loss import VectorRegressionEvidentialLoss
        return VectorRegressionEvidentialLoss(**loss_args)
    elif name == 'UncertaintyMSELoss':
        from uncertainty_ai.loss.uncertainty_loss import UncertaintyMSELoss
        return UncertaintyMSELoss()
    elif name =='EnsembleMSELoss':
        from uncertainty_ai.loss.ensemble_loss import EnsembleMSELoss
        return EnsembleMSELoss()
    elif name =='EnsembleGaussianNLLLoss':
        from uncertainty_ai.loss.ensemble_loss import EnsembleGaussianNLLLoss
        return EnsembleGaussianNLLLoss()
    elif name =='EnsembleMSENLLLoss':
        from uncertainty_ai.loss.ensemble_loss import EnsembleMSENLLLoss
        return EnsembleMSENLLLoss(**loss_args)
    elif name =='MSEGaussianNLLLoss':
        from uncertainty_ai.loss.uncertainty_loss import MSEGaussianNLLLoss
        return MSEGaussianNLLLoss(**loss_args)
    else:
        raise NotImplementedError()

def get_model_args(model, args):
    if model == "PaiNN":
        from uncertainty_ai.models.painn import PaiNN

        default_args = {"num_atoms":1,"bond_feat_dim":1,"num_targets":1,
                "n_atom_basis":128, "n_interactions":3, "n_rbf":20, "cutoff":5.0,
                "activation":"silu", "epsilon":1e-8, "regress_forces": True, "readout":"sum",
                "use_pbc": True, "otf_graph": False, "use_alpha": True, "l_dim": 1 }
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
    elif model == "PaiNNDropout":
        from uncertainty_ai.models.painn.painndropout import PaiNNDropout

        default_args = {"num_atoms":1,"bond_feat_dim":1,"num_targets":1,
                "n_atom_basis":128, "n_interactions":3, "n_rbf":20, "cutoff":5.0,
                "activation":"silu", "epsilon":1e-8, "regress_forces": True, "readout":"sum",
                "use_pbc": True, "otf_graph": False, "dropout_method": "bernoulli", "dropout_p": 0.1}
        for k, v in args.items():
            default_args[k] = v
        return PaiNNDropout, default_args
    elif model == "PaiNNVBDDropout":
        from uncertainty_ai.models.painn.painndropoutedge import PaiNNVBDDropout

        default_args = {"num_atoms":1,"bond_feat_dim":1,"num_targets":1,
                "n_atom_basis":128, "n_interactions":3, "n_rbf":20, "cutoff":5.0,
                "activation":"silu", "epsilon":1e-8, "regress_forces": True, "readout":"sum",
                "use_pbc": True, "otf_graph": False, "vbd_method": "sbp"}
        for k, v in args.items():
            default_args[k] = v
        return PaiNNVBDDropout, default_args
    elif model == "PaiNNDropout2":
        from uncertainty_ai.models.painn.painndropout import PaiNNDropout2

        default_args = {"num_atoms":1,"bond_feat_dim":1,"num_targets":1,
                "n_atom_basis":128, "n_interactions":3, "n_rbf":20, "cutoff":5.0,
                "activation":"silu", "epsilon":1e-8, "regress_forces": True, "readout":"sum",
                "use_pbc": True, "otf_graph": False, "dropout_method": "bernoulli", "dropout_p": 0.1}
        for k, v in args.items():
            default_args[k] = v
        return PaiNNDropout2, default_args
    elif model == "PaiNNDropout2V2":
        from uncertainty_ai.models.painn.painndropout import PaiNNDropout2V2

        default_args = {"num_atoms":1,"bond_feat_dim":1,"num_targets":1,
                "n_atom_basis":128, "n_interactions":3, "n_rbf":20, "cutoff":5.0,
                "activation":"silu", "epsilon":1e-8, "regress_forces": True, "readout":"sum",
                "use_pbc": True, "otf_graph": False, "dropout_method": "bernoulli", "dropout_p": 0.1}
        for k, v in args.items():
            default_args[k] = v
        return PaiNNDropout2V2, default_args
    elif model == "PaiNNDropout22":
        from uncertainty_ai.models.painn.painndropout import PaiNNDropout22

        default_args = {"num_atoms":1,"bond_feat_dim":1,"num_targets":1,
                "n_atom_basis":128, "n_interactions":3, "n_rbf":20, "cutoff":5.0,
                "activation":"silu", "epsilon":1e-8, "regress_forces": True, "readout":"sum",
                "use_pbc": True, "otf_graph": False, "dropout_method": "bernoulli", "dropout_p": 0.1, "use_alpha": True, "l_dim": 1}
        for k, v in args.items():
            default_args[k] = v
        return PaiNNDropout22, default_args
    elif model == "PaiNNDropout22MER":
        from uncertainty_ai.models.painn.painndropout import PaiNNDropout22MER

        default_args = {"num_atoms":1,"bond_feat_dim":1,"num_targets":1,
                "n_atom_basis":128, "n_interactions":3, "n_rbf":20, "cutoff":5.0,
                "activation":"silu", "epsilon":1e-8, "regress_forces": True, "readout":"sum",
                "use_pbc": True, "otf_graph": False, "dropout_method": "bernoulli", "dropout_p": 0.1}
        for k, v in args.items():
            default_args[k] = v
        return PaiNNDropout22MER, default_args
    elif model == "PaiNNDropout22V2":
        from uncertainty_ai.models.painn.painndropout import PaiNNDropout22V2

        default_args = {"num_atoms":1,"bond_feat_dim":1,"num_targets":1,
                "n_atom_basis":128, "n_interactions":3, "n_rbf":20, "cutoff":5.0,
                "activation":"silu", "epsilon":1e-8, "regress_forces": True, "readout":"sum",
                "use_pbc": True, "otf_graph": False, "dropout_method": "bernoulli", "dropout_p": 0.1}
        for k, v in args.items():
            default_args[k] = v
        return PaiNNDropout22V2, default_args
    elif model == "PaiNNVBDDropout22":
        from uncertainty_ai.models.painn.painndropout import PaiNNVBDDropout22
        default_args = {"num_atoms":1,"bond_feat_dim":1,"num_targets":1,
                "n_atom_basis":128, "n_interactions":3, "n_rbf":20, "cutoff":5.0,
                "activation":"silu", "epsilon":1e-8, "regress_forces": True, "readout":"sum",
                "use_pbc": True, "otf_graph": False, "vbd_method": "sbp"}
        for k, v in args.items():
            default_args[k] = v
        return PaiNNVBDDropout22, default_args
    elif model[:16] == "PaiNNVBDDropout2":
        default_args = {"num_atoms":1,"bond_feat_dim":1,"num_targets":1,
                "n_atom_basis":128, "n_interactions":3, "n_rbf":20, "cutoff":5.0,
                "activation":"silu", "epsilon":1e-8, "regress_forces": True, "readout":"sum",
                "use_pbc": True, "otf_graph": False, "vbd_method": "sbp"}
        for k, v in args.items():
            default_args[k] = v
        return globals()[model], default_args
    
    elif model == "PaiNNDropout3Edge":
        from uncertainty_ai.models.painn.painndropoutedge import PaiNNDropout3Edge

        default_args = {"num_atoms":1,"bond_feat_dim":1,"num_targets":1,
                "n_atom_basis":128, "n_interactions":3, "n_rbf":20, "cutoff":5.0,
                "activation":"silu", "epsilon":1e-8, "regress_forces": True, "readout":"sum",
                "use_pbc": True, "otf_graph": False, "dropout_method": "gaussian_noise", "dropout_p": 0.01}
        for k, v in args.items():
            default_args[k] = v
        return PaiNNDropout3Edge, default_args
    elif model in ["MACE2", "MACEVBDDropout22", "MACE2MER"]:
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
            "MLP_irreps":"16x0e+16x1o",
            "correlation": 3,
            "regress_forces": True,  # whether to regress forces
            "use_pbc": False,  # whether to use periodic boundary conditions
            "otf_graph": False,  # whether to use on-the-fly graph generation
        }
        for k, v in args.items():
            default_args[k] = v
        if model == "MACE2":
            from uncertainty_ai.models.mace2.mace import MACE
            return MACE, default_args
        elif model == "MACEVBDDropout22":
            from uncertainty_ai.models.mace2.mace import MACEVBDDropout22
            return MACEVBDDropout22, default_args
        elif model == "MACE2MER":
            from uncertainty_ai.models.mace2.mace import MACEMER
            return MACEMER, default_args
    elif model[:11] == "MACEDropout":
        
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
            "MLP_irreps":"16x0e+16x1o",
            "correlation": 3,
            "regress_forces": True,  # whether to regress forces
            "use_pbc": False,  # whether to use periodic boundary conditions
            "otf_graph": False,  # whether to use on-the-fly graph generation
            "dropout_method": "bernoulli",
            "dropout_p": 0.1,}
        for k, v in args.items():
            default_args[k] = v
        if model == "MACEDropout":
            from uncertainty_ai.models.mace2.mace import MACEDropout
            return MACEDropout, default_args
        elif model == "MACEDropout22":
            from uncertainty_ai.models.mace2.mace import MACEDropout22
            return MACEDropout22, default_args
        elif model == "MACEDropout22MER":
            from uncertainty_ai.models.mace2.mace import MACEDropout22MER
            return MACEDropout22MER, default_args
        elif model == "MACEDropout22V2":
            from uncertainty_ai.models.mace2.mace import MACEDropout22V2
            return MACEDropout22V2, default_args
        elif model == "MACEDropout3Edge":
            from uncertainty_ai.models.mace2.mace import MACEDropout3Edge
            return MACEDropout3Edge, default_args
        else:
            raise NotImplementedError("Unknown MACE dropout model: {}".format(model))
    elif model in ["NequIP", "NequIPVBDDropout22", "NequIPMER", "NequIPDropout22MER"]:
        default_args = {
            "num_atoms":1,
            "bond_feat_dim":1,  # not used
            "num_targets":1,
            "r_max":5.0,
            "num_bessel":8,
            "num_polynomial_cutoff":5,
            "max_ell":3,
            "num_interactions":3,
            "parity": True,  # whether to use parity
            "hidden_irreps":"32x0o+32x0e+32x1o+32x1e",
            "MLP_irreps":"16x0o+16x0e+16x1o+16x1e",
            "regress_forces": True,  # whether to regress forces
            "use_pbc": False,  # whether to use periodic boundary conditions
            "otf_graph": False,  # whether to use on-the-fly graph generation
            "resnet": False,  # whether to use resnet
        }
        for k, v in args.items():
            default_args[k] = v
        if model == "NequIP":
            from uncertainty_ai.models.nequip.nequip import NequIP
        
            return NequIP, default_args
        elif model == "NequIPVBDDropout22":
            from uncertainty_ai.models.nequip.nequip import NequIPVBDDropout22
            return NequIPVBDDropout22, default_args
        elif model == "NequIPMER":
            from uncertainty_ai.models.nequip.nequip import NequIPMER
            return NequIPMER, default_args
        elif model == "NequIPDropout22MER":
            from uncertainty_ai.models.nequip.nequip import NequIPDropout22MER
            return NequIPDropout22MER, default_args
    elif model[:13] == "NequIPDropout":
        default_args = {
            "num_atoms":1,
            "bond_feat_dim":1,  # not used
            "num_targets":1,
            "r_max":5.0,
            "num_bessel":8,
            "num_polynomial_cutoff":5,
            "max_ell":3,
            "num_interactions":3,
            "parity": True,  # whether to use parity
            "hidden_irreps":"32x0o+32x0e+32x1o+32x1e",
            "MLP_irreps":"16x0o+16x0e+16x1o+16x1e",
            "regress_forces": True,  # whether to regress forces
            "use_pbc": False,  # whether to use periodic boundary conditions
            "otf_graph": False,  # whether to use on-the-fly graph generation
            "resnet": False,  # whether to use resnet
            "dropout_method": "bernoulli",
            "dropout_p": 0.1,}
        for k, v in args.items():
            default_args[k] = v
        if model == "NequIPDropout":
            from uncertainty_ai.models.nequip.nequip import NequIPDropout
            return NequIPDropout, default_args
        elif model == "NequIPDropout22":
            from uncertainty_ai.models.nequip.nequip import NequIPDropout22
            return NequIPDropout22, default_args
        else:
            raise NotImplementedError("Unknown NequIP dropout model: {}".format(model))
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
