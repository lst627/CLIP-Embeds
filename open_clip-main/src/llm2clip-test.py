import os, sys
os.environ['HF_HOME'] = '/your-path/hf_cache/'
from PIL import Image
import json
from tqdm import tqdm
import torch
import random
import argparse
import clip
import csv
import pandas as pd
import ast
from datasets import load_dataset
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import CLIPImageProcessor
from llm2vec import LLM2Vec
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
model_name_or_path = "microsoft/LLM2CLIP-Openai-L-14-336" # or /path/to/local/LLM2CLIP-Openai-L-14-336
model = AutoModel.from_pretrained(
    model_name_or_path, 
    torch_dtype=torch.float16,
    trust_remote_code=True).to('cuda').eval()

llm_model_name = 'microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned'
config = AutoConfig.from_pretrained(
    llm_model_name, trust_remote_code=True
)
llm_model = AutoModel.from_pretrained(llm_model_name, config=config,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct' #  Workaround for LLM2VEC
l2v = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

def eval(dataset, root_dir, dataset_name):
    prepositions = ['on', 'under', 'front', 'behind', 'left', 'right']
    opposite = {'on': 'under', 'under': 'on', 'front': 'behind', 'behind': 'front', 'left': 'right', 'right': 'left'}

    eval_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                    d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                    {'left': 0, 'right': 0, \
                                    'on': 0, 'under': 0,
                                    'in-front': 0, 'behind': 0} for d in dataset}

    for d in tqdm(dataset):
        image_name = os.path.join(root_dir, d["image_path"][5:])
        object1, object2 = (d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5])
        object1 = object1.replace("-", " ")
        object2 = object2.replace("-", " ")
        gold_prep = list(set(prepositions).intersection(set(d['caption_options'][0].split())))
        gold_oppo = opposite[gold_prep[0]]

        image = Image.open(image_name).convert('RGB')

        # The ground truth is always the first one
        options = [s for s in d['caption_options'] if gold_prep[0] in s.split() or gold_oppo in s.split()]

        input_pixels = processor(images=image, return_tensors="pt").pixel_values.to('cuda')

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.get_image_features(input_pixels)
            text_features = l2v.encode(options, convert_to_tensor=True).to('cuda')
            text_features = model.get_text_features(text_features)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)


        correct = 1 if text_probs[0][0] > text_probs[0][1] else 0

        eval_dict[(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5])][d['image_path'].split('/')[-1].split('_')[1]] = correct


    set_correct = 0
    lr_pair_correct, lr_individual_correct = 0, 0
    ou_pair_correct, ou_individual_correct = 0, 0
    fb_pair_correct, fb_individual_correct = 0, 0
            

    for obj_pair, correct_dict in eval_dict.items():
        if correct_dict['left'] and correct_dict['right']:
            lr_pair_correct += 1
        lr_individual_correct += correct_dict['left']
        lr_individual_correct += correct_dict['right']
        # A
        if correct_dict['under'] and correct_dict['on']:
            ou_pair_correct += 1
        ou_individual_correct += correct_dict['under'] 
        ou_individual_correct += correct_dict['on']
        # B
        if correct_dict['behind'] and correct_dict['in-front']:
            fb_pair_correct += 1
        fb_individual_correct += correct_dict['behind'] 
        fb_individual_correct += correct_dict['in-front']
        if sum(correct_dict.values()) == 4:
            set_correct += 1
    pair_correct = lr_pair_correct + ou_pair_correct + fb_pair_correct
    indiv_accuracy = lr_individual_correct + ou_individual_correct + fb_individual_correct
    total_count = len(dataset)
    indiv_accuracy = indiv_accuracy*100/(total_count)
    pair_accuracy = pair_correct*100/(total_count/2)
    set_accuracy = set_correct*100/(total_count/4)

    with open("evaluation_results_llm2clip.txt", "a") as f:
        f.write("Individual accuracy: {}\n".format(indiv_accuracy))
        f.write("Left Right Individual accuracy: {}\n".format(lr_individual_correct*100/(total_count/2)))
        f.write("On Under Individual accuracy: {}\n".format(ou_individual_correct*100/(total_count/2)))
        f.write("Front Back Individual accuracy: {}\n".format(fb_individual_correct*100/(total_count/2)))
        f.write("Left Right Pair accuracy: {}\n".format(lr_pair_correct*100/(total_count/4)))
        f.write("On Under Pair accuracy: {}\n".format(ou_pair_correct*100/(total_count/4)))
        f.write("Front Back Pair accuracy: {}\n".format(fb_pair_correct*100/(total_count/4)))
        f.write("Pair accuracy: {}\n".format(pair_accuracy))
        f.write("Set accuracy: {}\n".format(set_accuracy))

def eval_4(dataset, root_dir, dataset_name):
    prepositions = ['on', 'under', 'front', 'behind', 'left', 'right']
    opposite = {'on': 'under', 'under': 'on', 'front': 'behind', 'behind': 'front', 'left': 'right', 'right': 'left'}

    eval_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                    d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                    {'left': 0, 'right': 0, \
                                    'on': 0, 'under': 0,
                                    'in-front': 0, 'behind': 0} for d in dataset}

    for d in tqdm(dataset):
        image_name = os.path.join(root_dir, d["image_path"][5:])
        object1, object2 = (d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5])
        object1 = object1.replace("-", " ")
        object2 = object2.replace("-", " ")

        image = Image.open(image_name).convert('RGB')

        # The ground truth is always the first one
        options = d['caption_options']

        input_pixels = processor(images=image, return_tensors="pt").pixel_values.to('cuda')

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.get_image_features(input_pixels)
            text_features = l2v.encode(options, convert_to_tensor=True).to('cuda')
            text_features = model.get_text_features(text_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        correct = 1 if text_probs[0][0] > text_probs[0][1] and text_probs[0][0] > text_probs[0][2] and text_probs[0][0] > text_probs[0][3] else 0

        eval_dict[(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5])][d['image_path'].split('/')[-1].split('_')[1]] = correct


    set_correct = 0
    lr_pair_correct, lr_individual_correct = 0, 0
    ou_pair_correct, ou_individual_correct = 0, 0
    fb_pair_correct, fb_individual_correct = 0, 0
            

    for obj_pair, correct_dict in eval_dict.items():
        if correct_dict['left'] and correct_dict['right']:
            lr_pair_correct += 1
        lr_individual_correct += correct_dict['left']
        lr_individual_correct += correct_dict['right']
        # A
        if correct_dict['under'] and correct_dict['on']:
            ou_pair_correct += 1
        ou_individual_correct += correct_dict['under'] 
        ou_individual_correct += correct_dict['on']
        # B
        if correct_dict['behind'] and correct_dict['in-front']:
            fb_pair_correct += 1
        fb_individual_correct += correct_dict['behind'] 
        fb_individual_correct += correct_dict['in-front']
        if sum(correct_dict.values()) == 4:
            set_correct += 1
    pair_correct = lr_pair_correct + ou_pair_correct + fb_pair_correct
    indiv_accuracy = lr_individual_correct + ou_individual_correct + fb_individual_correct
    total_count = len(dataset)
    indiv_accuracy = indiv_accuracy*100/(total_count)
    pair_accuracy = pair_correct*100/(total_count/2)
    set_accuracy = set_correct*100/(total_count/4)

    with open("evaluation_results_llm2clip.txt", "a") as f:
        f.write("Individual accuracy: {}\n".format(indiv_accuracy))
        f.write("Left Right Individual accuracy: {}\n".format(lr_individual_correct*100/(total_count/2)))
        f.write("On Under Individual accuracy: {}\n".format(ou_individual_correct*100/(total_count/2)))
        f.write("Front Back Individual accuracy: {}\n".format(fb_individual_correct*100/(total_count/2)))
        f.write("Left Right Pair accuracy: {}\n".format(lr_pair_correct*100/(total_count/4)))
        f.write("On Under Pair accuracy: {}\n".format(ou_pair_correct*100/(total_count/4)))
        f.write("Front Back Pair accuracy: {}\n".format(fb_pair_correct*100/(total_count/4)))
        f.write("Pair accuracy: {}\n".format(pair_accuracy))
        f.write("Set accuracy: {}\n".format(set_accuracy))

def eval_COCO_VG(dataset, root_dir, dataset_name):
    prepositions = ['top', 'bottom', 'above', 'below', 'left', 'right', 'front', 'behind']
    opposite = {'left': 'right', 'right': 'left', 'above': 'below', 'below': 'above', 'top': 'bottom', 'bottom': 'top', 'front': 'behind', 'behind': 'front'}

    eval_dict = {'left': 0, 'right': 0, \
            'top': 0, 'bottom': 0,
            'below': 0, "above": 0,
            'front': 0, 'behind': 0}  #####
    total_dict = {'left': 0, 'right': 0, \
                'top': 0, 'bottom': 0,
                'below': 0, "above": 0,
                'front': 0, 'behind': 0}  #####
    
    for d in tqdm(dataset):
        if "coco" in annotation_file:
            image = Image.open(os.path.join(root_dir, 'val2017/{}.jpg'.format(str(d[0]).zfill(12)))).convert('RGB')
        else:
            image = Image.open(os.path.join(root_dir, 'vg_images/{}.jpg'.format(d[0]))).convert('RGB')
        
        caption = d[1]

        gold_prep = list(set(prepositions).intersection(set(caption.split())))
        
        # The ground truth is always the first one
        options = [d[1], d[2]]

        input_pixels = processor(images=image, return_tensors="pt").pixel_values.to('cuda')

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.get_image_features(input_pixels)
            text_features = l2v.encode(options, convert_to_tensor=True).to('cuda')
            text_features = model.get_text_features(text_features)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)


        correct = 1 if text_probs[0][0] > text_probs[0][1] else 0

        eval_dict[gold_prep[0]] += correct
        total_dict[gold_prep[0]] += 1

    total_count = sum(total_dict.values())
    correct_count = sum(eval_dict.values())
    with open("evaluation_results_llm2clip.txt", "a") as f:
        f.write("Individual accuracy: {}\n".format(correct_count*100/total_count))
        f.write("Left Right Individual accuracy: {}\n".format((eval_dict["left"]+eval_dict['right'])*100/(total_dict["left"]+total_dict['right'])))
        if total_dict["top"]+total_dict['bottom'] > 0:
            f.write("Top Bottom Individual accuracy: {}\n".format((eval_dict["top"]+eval_dict['bottom'])*100/(total_dict["top"]+total_dict['bottom'])))
        if total_dict["above"]+total_dict['below'] > 0:
            f.write("Above Below Individual accuracy: {}\n".format((eval_dict["above"]+eval_dict['below'])*100/(total_dict["above"]+total_dict['below'])))
        if total_dict["front"]+total_dict['behind'] > 0:
            f.write("Front Behind Individual accuracy: {}\n".format((eval_dict["front"]+eval_dict['behind'])*100/(total_dict["front"]+total_dict['behind'])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on dataset')
    parser.add_argument('--dataset', type=str, default="a", help='Path to the annotation file')
    parser.add_argument('--root-dir', type=str, default="/your-path/whatsupdata/aro", help='Root directory of the dataset')
    
    args = parser.parse_args()

    if args.dataset == "a" or args.dataset == "a4":
        annotation_file = os.path.join(args.root_dir, "controlled_images_dataset.json") 
    elif args.dataset == "b" or args.dataset == "b4":
        annotation_file = os.path.join(args.root_dir, "controlled_clevr_dataset.json") 
    elif args.dataset == "cocoone":
        annotation_file = os.path.join(args.root_dir, "coco_qa_one_obj.json") 
    elif args.dataset == "cocotwo":
        annotation_file = os.path.join(args.root_dir, "coco_qa_two_obj.json")
    elif args.dataset == "vgone":
        annotation_file = os.path.join(args.root_dir, "vg_qa_one_obj.json") 
    elif args.dataset == "vgtwo":
        annotation_file = os.path.join(args.root_dir, "vg_qa_two_obj.json")

    
    with open("evaluation_results_llm2clip.txt", "a") as f:
        f.write("Model path: {} ".format(model_name_or_path))
        f.write("Dataset: {}\n".format(args.dataset))
    
    dataset = json.load(open(annotation_file))
    if args.dataset in ["a", "b"]: 
        eval(dataset, args.root_dir, args.dataset)
    elif args.dataset in ["a4", "b4"]: 
        eval_4(dataset, args.root_dir, args.dataset)
    else:
        eval_COCO_VG(dataset, args.root_dir, args.dataset)
            