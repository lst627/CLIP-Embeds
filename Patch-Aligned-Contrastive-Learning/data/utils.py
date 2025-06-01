import torchvision.transforms as T
import os, sys
import numpy as np
os.environ['HF_HOME'] = '/your-path/hf_cache/'
sys.path.append('/your-folder-path/CLIP-Embeds')
import torch
import open_clip.src.open_clip as oc

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

class prepare_data():
    def __init__(self, base_model='ViT-B-16'):
        if base_model == "ViT-L-14-336":
            self.val_transform = T.Compose([
                        T.ToTensor(),
                        T.Resize((336, 336)),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
            self.tokenizer = oc.get_tokenizer('ViT-L-14-336')
        else:
            self.val_transform = T.Compose([
                            T.ToTensor(),
                            T.Resize((400, 400)),
                            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])
            self.tokenizer = oc.get_tokenizer('ViT-B-16')

    def preprocess_image(self, image):
        if isinstance(image, list):
            image = torch.stack([self.val_transform(img) for img in image])
        else:
            image = self.val_transform(image)
        return image

    def preprocess_text(self,caption):
        return self.tokenizer(caption)
    
class prepare_data_clip():
    def __init__(self, image_processor):
        self.val_transform = image_processor
        self.tokenizer = oc.get_tokenizer('ViT-L-14-336')

    def preprocess_image(self, image):
        if isinstance(image, list):
            image = torch.stack([self.val_transform(img) for img in image])
        else:
            image = self.val_transform(image)
        return image
            
    def preprocess_text(self, caption):
        return self.tokenizer(caption)
    

class prepare_data_llm2clip():
    def __init__(self, img_size=336):
        self.val_transform = T.Compose([
                        T.ToTensor(),
                        T.Resize((img_size, img_size)),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])

    def preprocess_image(self, image):
        if isinstance(image, list):
            image = torch.stack([self.val_transform(img) for img in image])
        else:
            image = self.val_transform(image)
        return image
        

def get_scores(scores):
    """
    Calculate various scores based on the given results.

    Args:
        scores (dict or list): A dictionary or list containing results where each result can be:
            - dict: {id: {"q0_i0": 1 or 0, "q0_i1": 1 or 0, "q1_i0": 1 or 0, "q1_i1": 1 or 0}, ...}
            - list: [[q0_i0 (1 or 0), q0_i1 (1 or 0), q1_i0 (1 or 0), q1_i1 (1 or 0)], ...]

    The keys "q0_i0", "q0_i1", "q1_i0", "q1_i1" represent combinations of questions and images:
        - "q0_i0" means question_0 on image_0 
        - "q0_i1" means question_0 on image_1 
        - "q1_i0" means question_1 on image_0 
        - "q1_i1" means question_1 on image_1 

    Returns:
        dict: A dictionary containing the calculated scores:
            - 'question_score': Average question score
            - 'image_score': Average image score
            - 'binary_score': Average binary VQA score
            - 'group_score': Average group score
    """
    question_score = 0.0
    image_score = 0.0
    binary_score = 0.0
    group = 0.0

    num_samples = len(scores)

    def calculate_image_score(result):
        image_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q1_i0"] == 0.0:
                image_correct += 1
            if result["q1_i1"] == 1.0 and result["q0_i1"] == 0.0:
                image_correct += 1
        elif isinstance(result, list):
            if result[0] == 1.0 and result[2] == 0.0:
                image_correct += 1
            if result[3] == 1.0 and result[1] == 0.0:
                image_correct += 1
        return image_correct
    
    def calculate_question_score(result):
        text_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q0_i1"] == 0.0:
                text_correct += 1
            if result["q1_i1"] == 1.0 and result["q1_i0"] == 0.0:
                text_correct += 1
        else:
            if result[0] == 1.0 and result[1] == 0.0:
                text_correct += 1
            if result[3] == 1.0 and result[2] == 0.0:
                text_correct += 1
        return text_correct

    def calculate_binary_score(result):
        binary_score_correct = 0
        if isinstance(result, dict):
            binary_score_correct += 1 if result["q0_i0"] == 1.0 else 0
            binary_score_correct += 1 if result["q0_i1"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i0"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i1"] == 1.0 else 0
        else:
            binary_score_correct += 1 if result[0] == 1.0 else 0
            binary_score_correct += 1 if result[1] == 0.0 else 0
            binary_score_correct += 1 if result[2] == 0.0 else 0
            binary_score_correct += 1 if result[3] == 1.0 else 0

        return binary_score_correct

    def calculate_group(result):
        group_correct = 0
        if calculate_question_score(result) == 2 and calculate_image_score(result) == 2:
            group_correct += 1
        
        return group_correct
    
    if isinstance(scores, dict):
        for _, result in scores.items():
            question_score += calculate_question_score(result)
            image_score += calculate_image_score(result)
            binary_score += calculate_binary_score(result)
            group += calculate_group(result)
    else:
        for result in scores:
            question_score += calculate_question_score(result)
            image_score += calculate_image_score(result)
            binary_score += calculate_binary_score(result)
            group += calculate_group(result)

    results = {
        'question_score': question_score / float(num_samples * 2),
        'image_score': image_score / float(num_samples * 2),
        'binary_score': binary_score / float(num_samples * 4),
        'group_score': group / num_samples
    }

    return results