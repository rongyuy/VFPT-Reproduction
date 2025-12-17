export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1

# python tune_vtab.py \
#       --train-type "fpt" \
#       --config-file configs/prompt/prompt_fourier/Natural/cifar100_vpt_baseline.yaml \
#       MODEL.PROMPT_FOURIER.DEEP "True" \
#       MODEL.PROMPT_FOURIER.NUM_TOKENS "4" \
#       MODEL.PROMPT_FOURIER.DROPOUT "0.10" \
#       MODEL.PROMPT_FOURIER.FOURIER_PERCENTAGE "0.0" \
#       OUTPUT_DIR "./output_experiment/cifar100_vpt_baseline" \
#       DATA.BATCH_SIZE "128" \
#       MODEL.MODEL_ROOT "./models"

# python tune_vtab.py \
#       --train-type "fpt" \
#       --config-file configs/prompt/prompt_fourier/Specialized/eurosat_vpt_baseline.yaml \
#       MODEL.PROMPT_FOURIER.DEEP "True" \
#       MODEL.PROMPT_FOURIER.NUM_TOKENS "4" \
#       MODEL.PROMPT_FOURIER.DROPOUT "0.10" \
#       MODEL.PROMPT_FOURIER.FOURIER_PERCENTAGE "0.0" \
#       OUTPUT_DIR "./output_experiment/eurosat_vpt_baseline" \
#       DATA.BATCH_SIZE "128" \
#       MODEL.MODEL_ROOT "./models"

python tune_vtab.py \
      --train-type "fpt" \
      --config-file configs/prompt/prompt_fourier/Structured/clevr_distance_vpt_baseline.yaml \
      MODEL.PROMPT_FOURIER.DEEP "True" \
      MODEL.PROMPT_FOURIER.NUM_TOKENS "4" \
      MODEL.PROMPT_FOURIER.DROPOUT "0.10" \
      MODEL.PROMPT_FOURIER.FOURIER_PERCENTAGE "0.0" \
      OUTPUT_DIR "./output_experiment/clevr_vpt_baseline" \
      DATA.BATCH_SIZE "128" \
      MODEL.MODEL_ROOT "./models"