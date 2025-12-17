#!/bin/bash

# 1. 环境配置
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1

echo "========================================================"
echo "   Running Ablation Study: VFPT (FFT) + Gating          "
echo "========================================================"

# -------------------- CIFAR-100 --------------------
echo "[1/3] Running CIFAR-100 (FFT + Gating)..."
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
    MODEL.PROMPT_FOURIER.FOURIER_MODE "fft" \
    MODEL.PROMPT_FOURIER.USE_GATING "True" \
    \
    OUTPUT_DIR "./output_experiment/cifar100_vfpt_gating_only"

# -------------------- EuroSAT --------------------
echo "[2/3] Running EuroSAT (FFT + Gating)..."
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
    MODEL.PROMPT_FOURIER.FOURIER_MODE "fft" \
    MODEL.PROMPT_FOURIER.USE_GATING "True" \
    \
    OUTPUT_DIR "./output_experiment/eurosat_vfpt_gating_only"

# -------------------- CLEVR --------------------
echo "[3/3] Running CLEVR (FFT + Gating)..."
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
    MODEL.PROMPT_FOURIER.FOURIER_MODE "fft" \
    MODEL.PROMPT_FOURIER.USE_GATING "True" \
    \
    OUTPUT_DIR "./output_experiment/clevr_vfpt_gating_only"