#!/bin/bash
# Usage: bash train_scripts/train_qm7x.sh <painn|nequip|mace> [method ...]
# Runs configs/qm7x/<backbone>/<method>.yml (all methods if none are given).
BACKBONE=${1:?usage: $0 <painn|nequip|mace> [method ...]}
shift
METHODS=("$@")
if [ ${#METHODS[@]} -eq 0 ]; then
    METHODS=(backbone drop mer mer_d emer emer_d)
fi

for m in "${METHODS[@]}"; do
    echo "Running: configs/qm7x/$BACKBONE/$m.yml"
    python parallel_lightning.py --config "configs/qm7x/$BACKBONE/$m.yml"
done
echo "All jobs completed!"
