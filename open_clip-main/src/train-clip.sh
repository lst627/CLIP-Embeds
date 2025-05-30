# specify which GPUs you want to use.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export HF_HOME="/your-path/"

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node 4 -m training.main \
    --batch-size 128 \
    --precision amp \
    --workers 4 \
    --report-to tensorboard \
    --save-frequency 5 \
    --logs="/your-log-path/" \
    --dataset-type datamix \
    --warmup 50 \
    --lr=5e-6 \
    --wd=0.0 \
    --epochs=5 \
    --model ViT-L-14-336 \
    --pretrained 'openai' \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --lock-image 

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node 4 -m training.main \
    --batch-size 128 \
    --precision amp \
    --workers 4 \
    --report-to tensorboard \
    --save-frequency 5 \
    --logs="/your-log-path/" \
    --dataset-type datamix \
    --warmup 50 \
    --lr=5e-6 \
    --wd=0.0 \
    --epochs=5 \
    --model ViT-L-14-336 \
    --pretrained 'openai' \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --lock-image \
    --usehardtext \
    --augfiles leftright.json 
    