#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1 

# =========================================================
# 实验 1: CIFAR-100 (Natural) - VFPT + DCT + Gating
# =========================================================
echo "Running CIFAR-100 (DCT + Gating)..."
python tune_vtab.py \
    --train-type "fpt" \
    --config-file configs/prompt/prompt_fourier/Natural/cifar100_vpt_baseline.yaml \
    MODEL.PROMPT_FOURIER.DEEP "True" \
    MODEL.PROMPT_FOURIER.NUM_TOKENS "4" \
    MODEL.PROMPT_FOURIER.DROPOUT "0.10" \
    DATA.BATCH_SIZE "128" \
    MODEL.MODEL_ROOT "./models" \
    \
    MODEL.PROMPT_FOURIER.FOURIER_PERCENTAGE "1.0" \
    MODEL.PROMPT_FOURIER.FOURIER_MODE "dct" \
    MODEL.PROMPT_FOURIER.USE_GATING "True" \
    \
    OUTPUT_DIR "./output_experiment/cifar100_vfpt_dct_gating"

# =========================================================
# 实验 2: EuroSAT (Specialized) - VFPT + DCT + Gating
# =========================================================
echo "Running EuroSAT (DCT + Gating)..."
python tune_vtab.py \
    --train-type "fpt" \
    --config-file configs/prompt/prompt_fourier/Specialized/eurosat_vpt_baseline.yaml \
    MODEL.PROMPT_FOURIER.DEEP "True" \
    MODEL.PROMPT_FOURIER.NUM_TOKENS "4" \
    MODEL.PROMPT_FOURIER.DROPOUT "0.10" \
    DATA.BATCH_SIZE "128" \
    MODEL.MODEL_ROOT "./models" \
    \
    MODEL.PROMPT_FOURIER.FOURIER_PERCENTAGE "1.0" \
    MODEL.PROMPT_FOURIER.FOURIER_MODE "dct" \
    MODEL.PROMPT_FOURIER.USE_GATING "True" \
    \
    OUTPUT_DIR "./output_experiment/eurosat_vfpt_dct_gating"

# =========================================================
# 实验 3: CLEVR (Structured) - VFPT + DCT + Gating
# =========================================================
echo "Running CLEVR (DCT + Gating)..."
python tune_vtab.py \
    --train-type "fpt" \
    --config-file configs/prompt/prompt_fourier/Structured/clevr_distance_vpt_baseline.yaml \
    MODEL.PROMPT_FOURIER.DEEP "True" \
    MODEL.PROMPT_FOURIER.NUM_TOKENS "4" \
    MODEL.PROMPT_FOURIER.DROPOUT "0.10" \
    DATA.BATCH_SIZE "128" \
    MODEL.MODEL_ROOT "./models" \
    \
    MODEL.PROMPT_FOURIER.FOURIER_PERCENTAGE "1.0" \
    MODEL.PROMPT_FOURIER.FOURIER_MODE "dct" \
    MODEL.PROMPT_FOURIER.USE_GATING "True" \
    \
    OUTPUT_DIR "./output_experiment/clevr_vfpt_dct_gating"