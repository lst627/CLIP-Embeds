export HF_HOME="/your-path/hf_cache/"
export CUDA_VISIBLE_DEVICES=0

python eval_clip.py --dataset=a
python eval_clip.py --dataset=b
python eval_clip.py --dataset=cocoone
python eval_clip.py --dataset=cocotwo
python eval_clip.py --dataset=vgone
python eval_clip.py --dataset=vgtwo
python eval_clip.py --dataset=a4
python eval_clip.py --dataset=b4
python eval_clip.py --dataset=mmvpvlm --root-dir="/your-path/MMVP_VLM/" 
python eval_clip.py --dataset=mmvp --root-dir="/your-path/MMVP/" 

for w in pacl_all.pth pacl_rope_all.pth 
do
    python eval_pacl.py --dataset=a --model-path=$w
    python eval_pacl.py --dataset=b --model-path=$w 
    python eval_pacl.py --dataset=cocoone --model-path=$w 
    python eval_pacl.py --dataset=cocotwo --model-path=$w 
    python eval_pacl.py --dataset=vgone --model-path=$w 
    python eval_pacl.py --dataset=vgtwo --model-path=$w 
    python eval_pacl.py --dataset=a4 --model-path=$w 
    python eval_pacl.py --dataset=b4 --model-path=$w 
done

for w in pacl_rope_llm_all.pth 
do
    python eval_llm2pacl.py --dataset=a --model-path=$w
    python eval_llm2pacl.py --dataset=b --model-path=$w 
    python eval_llm2pacl.py --dataset=cocoone --model-path=$w 
    python eval_llm2pacl.py --dataset=cocotwo --model-path=$w 
    python eval_llm2pacl.py --dataset=vgone --model-path=$w 
    python eval_llm2pacl.py --dataset=vgtwo --model-path=$w 
    python eval_llm2pacl.py --dataset=a4 --model-path=$w 
    python eval_llm2pacl.py --dataset=b4 --model-path=$w 
done

for w in sparc.pth sparc_rope.pth 
do
    python eval_sparc.py --dataset=a --model-path=$w --local
    python eval_sparc.py --dataset=b --model-path=$w --local
    python eval_sparc.py --dataset=cocoone --model-path=$w --local
    python eval_sparc.py --dataset=cocotwo --model-path=$w --local
    python eval_sparc.py --dataset=vgone --model-path=$w --local
    python eval_sparc.py --dataset=vgtwo --model-path=$w --local
    python eval_sparc.py --dataset=a4 --model-path=$w --local
    python eval_sparc.py --dataset=b4 --model-path=$w --local
done

python eval_vqa_score.py --dataset=a
python eval_vqa_score.py --dataset=b
python eval_vqa_score.py --dataset=cocoone
python eval_vqa_score.py --dataset=cocotwo
python eval_vqa_score.py --dataset=vgone
python eval_vqa_score.py --dataset=vgtwo
python eval_vqa_score.py --dataset=a4
python eval_vqa_score.py --dataset=b4
python eval_vqa_score.py --dataset=mmvpvlm --root-dir="/your-path/MMVP_VLM/" 
python eval_vqa_score.py --dataset=mmvp --root-dir="/your-path/MMVP/" 