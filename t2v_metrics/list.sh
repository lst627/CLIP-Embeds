export HF_HOME="/your-path/hf_cache/"
export CUDA_VISIBLE_DEVICES=0
python eval.py --model openai:ViT-L-14-336
python eval.py --model llava-v1.5-7b
python eval.py --model llava-llama-3
python eval.py --model llava-phi-3
python eval.py --model laion400m_s11b_b41k:EVA01-g-14