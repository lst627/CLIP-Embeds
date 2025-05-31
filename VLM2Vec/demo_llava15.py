
import os
os.environ['HF_HOME'] = "/your-path/hf_cache/"
from src.model import MMEBModel
from src.arguments import ModelArguments
from src.utils import load_processor
import torch
from transformers import HfArgumentParser, AutoProcessor
from PIL import Image
import numpy as np

model_args = ModelArguments(
    model_name='llava-hf/llava-1.5-7b-hf',
    checkpoint_path='/your-path/vlm2vec_llava_1.5/checkpoint-900',
    pooling='last',
    normalize=True,
    lora=True, 
    lora_r=8,
    model_backbone='llava_1.5',
    num_crops=16)

processor = load_processor(model_args)

model = MMEBModel.load(model_args)
model.eval()
model = model.to('cuda', dtype=torch.bfloat16)


# Image + Text -> Text
# print(processor)
inputs = processor('<image> Represent the given image with the following question: What is in the image', [Image.open(
    'figures/example.jpg')], return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
qry_output = model(qry=inputs)["qry_reps"]

string = 'A cat and a dog'
inputs = processor(string, return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))

string = 'A cat and a tiger'
inputs = processor(string, return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))

# Text -> Image
inputs = processor('Find me an everyday image that matches the given caption: A cat and a dog.', return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
qry_output = model(qry=inputs)["qry_reps"]

string = '<image> Represent the given image.'
inputs = processor(string, [Image.open('figures/example.jpg')], return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))

inputs = processor('Find me an everyday image that matches the given caption: A cat and a tiger.', return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
qry_output = model(qry=inputs)["qry_reps"]

string = '<image> Represent the given image.'
inputs = processor(string, [Image.open('figures/example.jpg')], return_tensors="pt")
inputs = {key: value.to('cuda') for key, value in inputs.items()}
tgt_output = model(tgt=inputs)["tgt_reps"]
print(string, '=', model.compute_similarity(qry_output, tgt_output))
