#!/bin/bash
# This script will train ViT-VQ-AE inside runai job

echo "Launching final ViT-VQ-AE training..."
cd /mnt/mahdipou/nsd/experiment2/vitvq/final

source activate senv  # یا conda activate senv
python train_fmri.py config.yaml
