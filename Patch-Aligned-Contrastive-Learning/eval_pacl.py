import torch
import torch.nn as nn
import torchvision.transforms as T
import os 
os.environ['HF_HOME'] = '/your-path/hf_cache/'
from data.utils import prepare_data
from model.pacl import open_clip_pacl, open_clip_pacl_rope, open_clip_pacl_rope_after
from PIL import Image
import argparse
import json
from tqdm import tqdm
import csv

"""
prepare model
"""
device = torch.device("cuda:0")

process = prepare_data(base_model = "ViT-L-14-336")


print("\n\n\n===================================================================================\n\n")



def eval(model, dataset, root_dir, dataset_name):
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

        input_pixels = process.preprocess_image(image).unsqueeze(0).to(device)
        tokenized_text = process.preprocess_text(options).to(device)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Be sure to check which entries should be compared !!!!!
            image_features, text_features = model(input_pixels, tokenized_text)
            text_probs = 100.0 * image_features @ text_features.T
            correct = 1 if text_probs[0][0] > text_probs[1][1] else 0

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

    with open("evaluation_results.txt", "a") as f:
        f.write("Individual accuracy: {}\n".format(indiv_accuracy))
        f.write("Left Right Individual accuracy: {}\n".format(lr_individual_correct*100/(total_count/2)))
        f.write("On Under Individual accuracy: {}\n".format(ou_individual_correct*100/(total_count/2)))
        f.write("Front Back Individual accuracy: {}\n".format(fb_individual_correct*100/(total_count/2)))
        f.write("Left Right Pair accuracy: {}\n".format(lr_pair_correct*100/(total_count/4)))
        f.write("On Under Pair accuracy: {}\n".format(ou_pair_correct*100/(total_count/4)))
        f.write("Front Back Pair accuracy: {}\n".format(fb_pair_correct*100/(total_count/4)))
        f.write("Pair accuracy: {}\n".format(pair_accuracy))
        f.write("Set accuracy: {}\n".format(set_accuracy))

def eval_4(model, dataset, root_dir, dataset_name):
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
        options = d['caption_options']

        input_pixels = process.preprocess_image(image).unsqueeze(0).to(device)
        tokenized_text = process.preprocess_text(options).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Be sure to check which entries should be compared !!!!!
            image_features, text_features = model(input_pixels, tokenized_text)
            text_probs = 100.0 * image_features @ text_features.T
            correct = 1 if text_probs[0][0] > text_probs[1][1] and text_probs[0][0] > text_probs[2][2] and text_probs[0][0] > text_probs[3][3] else 0

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

    with open("evaluation_results.txt", "a") as f:
        f.write("Individual accuracy: {}\n".format(indiv_accuracy))
        f.write("Left Right Individual accuracy: {}\n".format(lr_individual_correct*100/(total_count/2)))
        f.write("On Under Individual accuracy: {}\n".format(ou_individual_correct*100/(total_count/2)))
        f.write("Front Back Individual accuracy: {}\n".format(fb_individual_correct*100/(total_count/2)))
        f.write("Left Right Pair accuracy: {}\n".format(lr_pair_correct*100/(total_count/4)))
        f.write("On Under Pair accuracy: {}\n".format(ou_pair_correct*100/(total_count/4)))
        f.write("Front Back Pair accuracy: {}\n".format(fb_pair_correct*100/(total_count/4)))
        f.write("Pair accuracy: {}\n".format(pair_accuracy))
        f.write("Set accuracy: {}\n".format(set_accuracy))

def eval_COCO_VG(model, dataset, root_dir, dataset_name):
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
        # A photo of a xxx on the left
        # A photo of a xxx   to the left of a xxx / above a xxx (COCO)
        # A photo of a xxx to the behind of a xxx (VG two)

        gold_prep = list(set(prepositions).intersection(set(caption.split())))
        gold_oppo = opposite[gold_prep[0]]
        
        # The ground truth is always the first one
        options = [d[1], d[2]]

        input_pixels = process.preprocess_image(image).unsqueeze(0).to(device)
        tokenized_text = process.preprocess_text(options).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Be sure to check which entries should be compared !!!!!
            image_features, text_features = model(input_pixels, tokenized_text)
            text_probs = 100.0 * image_features @ text_features.T
            # text_probs = text_probs.softmax(dim=-1)
            correct = 1 if text_probs[0][0] > text_probs[1][1] else 0

        eval_dict[gold_prep[0]] += correct
        total_dict[gold_prep[0]] += 1

    total_count = sum(total_dict.values())
    correct_count = sum(eval_dict.values())
    with open("evaluation_results.txt", "a") as f:
        f.write("Individual accuracy: {}\n".format(correct_count*100/total_count))
        f.write("Left Right Individual accuracy: {}\n".format((eval_dict["left"]+eval_dict['right'])*100/(total_dict["left"]+total_dict['right'])))
        if total_dict["top"]+total_dict['bottom'] > 0:
            f.write("Top Bottom Individual accuracy: {}\n".format((eval_dict["top"]+eval_dict['bottom'])*100/(total_dict["top"]+total_dict['bottom'])))
        if total_dict["above"]+total_dict['below'] > 0:
            f.write("Above Below Individual accuracy: {}\n".format((eval_dict["above"]+eval_dict['below'])*100/(total_dict["above"]+total_dict['below'])))
        if total_dict["front"]+total_dict['behind'] > 0:
            f.write("Front Behind Individual accuracy: {}\n".format((eval_dict["front"]+eval_dict['behind'])*100/(total_dict["front"]+total_dict['behind'])))


def eval_MMVP(model, root_dir, dataset_name):
    evaluate_mode = "t2i"
    if dataset_name == "mmvpvlm":
        image_dir = os.path.join(root_dir, 'MLLM_VLM_Images')
        csv_file = os.path.join(root_dir, 'Questions.csv')
        categories = [
            'Orientation and Direction', 'Presence of Specific Features', 
            'State and Condition', 'Quantity and Count', 
            'Positional and Relational Context', 'Color and Appearance',
            'Structural Characteristics', 'Texts',
            'Viewpoint and Perspective'
        ]
        # additional question
        question_file = os.path.join(root_dir, 'Questions-llava.csv')
        with open(question_file, 'r') as qf:
            question_reader = csv.reader(qf)
            next(question_reader)  # skip header
    else: # "mmvp"
        image_dir = os.path.join(root_dir, 'MMVP_Images')
        csv_file = os.path.join(root_dir, 'Questions-clip.csv')
        categories = [
            'Unknown'
        ]
        question_file = os.path.join(root_dir, 'Questions.csv')
        with open(question_file, 'r') as qf:
            question_reader = csv.reader(qf)
            next(question_reader)  # skip header

    csv_outfile = open('output.csv', 'w', newline='')
    csv_writer = csv.writer(csv_outfile)
    csv_writer.writerow(['qid1', 'qid2', 'pred1', 'pred2', 'gt1', 'gt2', 'q1score', 'q2score'])  # header

    pair_accuracies = {category: 0 for category in categories}
    single_accuracies = {category: 0 for category in categories}
    num_pairs = 0
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            qid1, qtype1, statement1 = row
        
            # Get next row for the pair
            row = next(reader, None)
            if not row:
                break
            qid2, qtype2, statement2 = row
            
            qid1, qid2 = int(qid1), int(qid2)
            
            if dataset_name == "mmvpvlm":
                img1 = Image.open(os.path.join(image_dir, qtype1, f'{qid1}.jpg'))
                img2 = Image.open(os.path.join(image_dir, qtype1, f'{qid2}.jpg'))
            else:
                img1 = Image.open(os.path.join(image_dir, f'{qid1}.jpg'))
                img2 = Image.open(os.path.join(image_dir, f'{qid2}.jpg'))
                # text1, text2 = (dl.iloc[i//2]["A"], dl.iloc[i//2]["B"])

            text1 = 'a photo of ' + statement1
            text2 = 'a photo of ' + statement2

            img1 = process.preprocess_image(img1).unsqueeze(0).to(device)
            img2 = process.preprocess_image(img2).unsqueeze(0).to(device)
            text1 = process.preprocess_text([text1]).to(device)
            text2 = process.preprocess_text([text2]).to(device)
            texts = torch.cat((text1, text2), dim=0)
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                # one image at a time!
                image_feature_1, text_features = model(img1, texts)
                text_probs_1 = 100.0 * image_feature_1 @ text_features.T   # 2 * 2
                image_feature_2, text_features = model(img2, texts)
                text_probs_2 = 100.0 * image_feature_2 @ text_features.T   # 2 * 2
                logits_per_text = torch.tensor([[text_probs_1[0][0], text_probs_1[1][1]], [text_probs_2[0][0], text_probs_2[1][1]]]).float().T
            
            probs1 = logits_per_text[0].softmax(dim=-1).cpu().numpy()
            probs2 = logits_per_text[1].softmax(dim=-1).cpu().numpy()
            img1_score1 = probs1[0]
            img1_score2 = probs2[0]
            
            pred1 = "img1" if img1_score1 > 0.5 else "img2"
            pred2 = "img1" if img1_score2 > 0.5 else "img2"
            
            gt1 = "img1" if qid1 % 2 == 1 else "img2"
            gt2 = "img1" if qid2 % 2 == 1 else "img2"
            csv_writer.writerow([qid1, qid2, pred1, pred2, gt1, gt2, img1_score1, img1_score2])
                
            if dataset_name == "mmvpvlm":
                current_category = categories[num_pairs // 15]
            else:
                current_category = categories[0]
            if pred1 == gt1 and pred2 == gt2:
                pair_accuracies[current_category] += 1
            if pred1 == gt1:
                single_accuracies[current_category] += 1
            if pred2 == gt2:
                single_accuracies[current_category] += 1
            num_pairs += 1

        csv_outfile.close()
    
    with open("evaluation_results.txt", "a") as f:
        f.write(f"Pair: {100*sum(pair_accuracies.values())/num_pairs}, Individual: {100*sum(single_accuracies.values())/num_pairs/2}\n")

    # Calculate percentage accuracies
    for category in pair_accuracies:
        pair_accuracies[category] = (pair_accuracies[category] / (num_pairs // len(categories))) * 100
        single_accuracies[category] = (single_accuracies[category] / (num_pairs*2 // len(categories))) * 100

    with open("evaluation_results.txt", "a") as f:
        for category, accuracy in pair_accuracies.items():
            f.write(f"{category} Pair accuracy: {accuracy}\n")
        for category, accuracy in single_accuracies.items():
            f.write(f"{category} Single accuracy: {accuracy}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on dataset')
    parser.add_argument('--model-path', type=str, default="pacl_ft.pth", help='Path to the model weight')
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

    base_model = "ViT-L-14-336" if "all" in args.model_path else "ViT-B-16"
    if "rope" in args.model_path:
        if "after" in args.model_path:
            model = open_clip_pacl_rope_after(base_model)
        else:
            model = open_clip_pacl_rope(base_model)
    else:
        model = open_clip_pacl(base_model)
    model_weights = model.state_dict()
    saved_weights = torch.load(args.model_path, map_location=device)
    for name in model_weights:
        model_weights[name] = saved_weights['module.' + name]
    for name in saved_weights:
        if name[7:] not in model_weights:
            print(name+" in weights is not included\n")
            raise ValueError
    model.load_state_dict(model_weights)
    for p in model.parameters(): p.requires_grad = False
    model.to(device)
    model.eval()

    with open("evaluation_results.txt", "a") as f:
        f.write("Model path: {} ".format(args.model_path))
        f.write("Dataset: {}\n".format(args.dataset))
    
    if args.dataset in ["mmvp", "mmvpvlm"]:
        eval_MMVP(model, args.root_dir, args.dataset)
    else:
        dataset = json.load(open(annotation_file))
        if args.dataset in ["a", "b"]: 
            eval(model, dataset, args.root_dir, args.dataset)
        elif args.dataset in ["a4", "b4"]: 
            eval_4(model, dataset, args.root_dir, args.dataset)
        elif args.dataset in ["cocoone", "cocotwo", "vgone", "vgtwo"]:
            eval_COCO_VG(model, dataset, args.root_dir, args.dataset)