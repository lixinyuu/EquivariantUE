#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time: 03/06/23 23:09:35
# File: dataset.py

from .md17 import MD17, MD17Aligned, RotatedDataset
from .qm7x import QM7X, QM9P
import torch

class FloatToTensorTransform:
    def __call__(self, data):
        # Assuming data.y is a float or an array of floats
        data.y = torch.tensor(data.y, dtype=torch.float32)
        return data