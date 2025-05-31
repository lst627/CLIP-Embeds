export PYTHONPATH=./:$PYTHONPATH  # Change to your own path

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --max_restarts=0 train.py \
  --model_name llava-hf/llava-1.5-7b-hf \
  --model_backbone llava_1.5 \
  --output_dir /your-path/vlm2vec_llava_1.5 \
  --bf16 --pooling last \
  --lora --lora_r 8 \
  --dataset_name CombinedDataset \
  --max_len 1024 --logging_steps 1 \
  --lr_scheduler_type linear --learning_rate 2e-5 --max_steps 1000 \
  --warmup_steps 100 --save_steps 100 --normalize True \
  --temperature 0.02 --per_device_train_batch_size 64 \
  --grad_cache True --gc_q_chunk_size 1 --gc_p_chunk_size 1

