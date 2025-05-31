export HF_HOME="/your-path/hf_cache/"
export CUDA_VISIBLE_DEVICES=0
python eval_llava15.py --dataset=a
python eval_llava15.py --dataset=b
python eval_llava15.py --dataset=cocoone
python eval_llava15.py --dataset=cocotwo
python eval_llava15.py --dataset=vgone
python eval_llava15.py --dataset=vgtwo
python eval_llava15.py --dataset=a4
python eval_llava15.py --dataset=b4
python eval_llava15.py --dataset=mmvpvlm --root-dir="/your-path/MMVP_VLM/" 
python eval_llava15.py --dataset=mmvp --root-dir="/your-path/MMVP/"