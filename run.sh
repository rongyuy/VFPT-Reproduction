export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

python tune_vtab.py \
      --train-type "fpt" \
      --config-file configs/prompt/prompt_fourier/Natural/cifar100_forVPT.yaml \
      MODEL.PROMPT_FOURIER.DEEP "True" \
      MODEL.PROMPT_FOURIER.NUM_TOKENS "4" \
      MODEL.PROMPT_FOURIER.DROPOUT "0.10" \
      MODEL.PROMPT_FOURIER.FOURIER_PERCENTAGE "1.0" \
      OUTPUT_DIR "./output_experiment/cifar100_gpu_test" \
      DATA.BATCH_SIZE "128" \
      MODEL.MODEL_ROOT "./models"

python tune_vtab.py \
      --train-type "fpt" \
      --config-file configs/prompt/prompt_fourier/Specialized/eurosat.yaml \
      MODEL.PROMPT_FOURIER.DEEP "True" \
      MODEL.PROMPT_FOURIER.NUM_TOKENS "4" \
      MODEL.PROMPT_FOURIER.DROPOUT "0.10" \
      MODEL.PROMPT_FOURIER.FOURIER_PERCENTAGE "1.0" \
      OUTPUT_DIR "./output_experiment/eurosat_test" \
      DATA.BATCH_SIZE "128" \
      MODEL.MODEL_ROOT "./models"

python tune_vtab.py \
      --train-type "fpt" \
      --config-file configs/prompt/prompt_fourier/Structured/clevr_distance.yaml \
      MODEL.PROMPT_FOURIER.DEEP "True" \
      MODEL.PROMPT_FOURIER.NUM_TOKENS "4" \
      MODEL.PROMPT_FOURIER.DROPOUT "0.10" \
      MODEL.PROMPT_FOURIER.FOURIER_PERCENTAGE "1.0" \
      OUTPUT_DIR "./output_experiment/clevr_test" \
      DATA.BATCH_SIZE "128" \
      MODEL.MODEL_ROOT "./models"