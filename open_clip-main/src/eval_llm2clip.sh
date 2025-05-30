export HF_HOME="/your-path/hf_cache/"
export CUDA_VISIBLE_DEVICES=1

python llm2clip-test.py --dataset=a
python llm2clip-test.py --dataset=b
python llm2clip-test.py --dataset=cocoone
python llm2clip-test.py --dataset=cocotwo
python llm2clip-test.py --dataset=vgone
python llm2clip-test.py --dataset=vgtwo
python llm2clip-test.py --dataset=a4
python llm2clip-test.py --dataset=b4