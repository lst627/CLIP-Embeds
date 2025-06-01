from torch.utils.data import Dataset
import torchvision.transforms as T
from pycocotools.coco import COCO
import os
from PIL import Image
import spacy
import json
import random
import sys
import numpy as np
sys.path.append('/your-folder-path/CLIP-Embeds')
import open_clip.src.open_clip as oc

class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file, apply_transform=False, img_size=400):

        # chunk for original COCO dataloader
        self.root_dir = root_dir
        self.train_transform = T.Compose([
                        T.Resize((img_size, img_size)),
                        T.ToTensor(),
                        # T.RandomHorizontalFlip(0.5),
                        T.ColorJitter(brightness=.2, hue=.1),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.val_transform = T.Compose([
                        T.Resize((img_size, img_size)),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.apply_transform = apply_transform
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        # chunk for noun phrase extraction -->> creating a prompt from template
        self.template = [
            'a picture of {}.',
            'itap of {}.',
            'a photograph of {}.',
            'this picture contains {}.',
            'a good photo of {}.'
        ]
        self.nlptk = spacy.load("en_core_web_sm")

        # chunk for tokenization
        self.open_clip_tokenizer = oc.get_tokenizer('ViT-B-16')


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        
        # chunk for original COCO dataloader
        coco = self.coco
        img_id = self.ids[index]
        caption = coco.imgToAnns[img_id][0]['caption'] # a python string

        img_info = coco.loadImgs(img_id)[0]
        img_path = f"{self.root_dir}/{img_info['file_name']}"
        image = Image.open(img_path).convert('RGB')
        if self.apply_transform == True:
            image = self.train_transform(image) # tensor of shape = [3, H, W]
        else:
            image = self.val_transform(image)

        # chunk for noun phrase extraction -->> creating a prompt from template
        processed_text = self.nlptk(caption) 
        all_noun_phrases = [chunk.text.lower() for chunk in processed_text.noun_chunks]
        random_template = random.choice(self.template)

        # use original caption 50% of the time
        nounphrase_or_full_caption = random.choice([0,1])

        if len(all_noun_phrases) != 0 and nounphrase_or_full_caption == 0:
            random_noun_phrase = random.choice(all_noun_phrases)
            single_noun_phrase_per_img = random_template.format(random_noun_phrase)
        else:
            single_noun_phrase_per_img = caption
        
        tokenized_phrase = self.open_clip_tokenizer(single_noun_phrase_per_img).squeeze()
        return image, tokenized_phrase

class LCS558KDataset(Dataset):
    def __init__(self, apply_transform=False, img_size=400, use_llm=False, embed_path=None):

        # chunk for original COCO dataloader
        self.root_dir = "/your-path/LLaVA-Pretrain"
        f2 = open(os.path.join(self.root_dir, "blip_laion_cc_sbu_558k.json"), "r")
        annotations = json.load(f2)
        f2.close()

        self.sample_list = []
        for sample in annotations:
            if 'image' in sample:
                self.sample_list.append(sample)

        self.train_transform = T.Compose([
                        T.Resize((img_size, img_size)),
                        T.ToTensor(),
                        # T.RandomHorizontalFlip(0.5),
                        T.ColorJitter(brightness=.2, hue=.1),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.val_transform = T.Compose([
                        T.Resize((img_size, img_size)),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.apply_transform = apply_transform

        # chunk for noun phrase extraction -->> creating a prompt from template
        self.template = [
            'a picture of {}.',
            'itap of {}.',
            'a photograph of {}.',
            'this picture contains {}.',
            'a good photo of {}.'
        ]
        self.nlptk = spacy.load("en_core_web_sm")

        # chunk for tokenization
        self.open_clip_tokenizer = oc.get_tokenizer('ViT-B-16')
        self.use_llm = use_llm
        self.embed = None
        # the first is template, and the second is full caption
        if embed_path != None:
            self.embed = []
            for p in embed_path:
                embed = np.load(p)
                self.embed.append(embed)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        # chunk for original COCO dataloader
        sample = self.sample_list[index]
        img_path = os.path.join(self.root_dir, "images", sample["image"])
        caption = sample["conversations"][1]['value'] # a python string
        image = Image.open(img_path).convert('RGB')
        if self.apply_transform == True:
            image = self.train_transform(image) # tensor of shape = [3, H, W]
        else:
            image = self.val_transform(image)        
        # use original caption 50% of the time
        nounphrase_or_full_caption = random.choice([0,1])
        # chunk for noun phrase extraction -->> creating a prompt from template
        processed_text = self.nlptk(caption) 
        all_noun_phrases = [chunk.text.lower() for chunk in processed_text.noun_chunks]
        random_template = random.choice(self.template)
        if self.use_llm:
            if len(all_noun_phrases) != 0 and nounphrase_or_full_caption == 0:
                tokenized_phrase = self.embed[0][index]
            else:
                tokenized_phrase = self.embed[1][index]
        else:
            if len(all_noun_phrases) != 0 and nounphrase_or_full_caption == 0:
                random_noun_phrase = random.choice(all_noun_phrases)
                single_noun_phrase_per_img = random_template.format(random_noun_phrase)
            else:
                single_noun_phrase_per_img = caption
            
            tokenized_phrase = self.open_clip_tokenizer(single_noun_phrase_per_img).squeeze()
        return image, tokenized_phrase

class DataMixDataset(Dataset):
    def __init__(self, apply_transform=False, img_size=400):

        # chunk for original COCO dataloader
        self.root_dir = "/your-path/datamix665k/"
        f1 = open("/your-path/LLaVA-Instruct-150K/llava_v1_5_mix665k.json", "r")
        annotations = json.load(f1)
        f1.close()

        self.sample_list = []
        for sample in annotations:
            if 'image' in sample:
                self.sample_list.append(sample)

        self.train_transform = T.Compose([
                        T.Resize((img_size, img_size)),
                        T.ToTensor(),
                        # T.RandomHorizontalFlip(0.5),
                        T.ColorJitter(brightness=.2, hue=.1),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.val_transform = T.Compose([
                        T.Resize((img_size, img_size)),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.apply_transform = apply_transform

        # chunk for noun phrase extraction -->> creating a prompt from template
        self.template = [
            'a picture of {}.',
            'itap of {}.',
            'a photograph of {}.',
            'this picture contains {}.',
            'a good photo of {}.'
        ]
        self.nlptk = spacy.load("en_core_web_sm")

        # chunk for tokenization
        self.open_clip_tokenizer = oc.get_tokenizer('ViT-B-16')


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        # chunk for original COCO dataloader
        sample = self.sample_list[index]
        img_path = os.path.join(self.root_dir, sample["image"])
        i = random.randint(0, len(sample["conversations"]) // 2 - 1)
        question = sample["conversations"][i*2]['value']
        caption = sample["conversations"][i*2+1]['value'] # a python string
        image = Image.open(img_path).convert('RGB')
        if self.apply_transform == True:
            image = self.train_transform(image) # tensor of shape = [3, H, W]
        else:
            image = self.val_transform(image)

        single_noun_phrase_per_img = caption
        tokenized_phrase = self.open_clip_tokenizer(single_noun_phrase_per_img).squeeze()
        tokenized_question = self.open_clip_tokenizer(question).squeeze()
        return image, tokenized_phrase, tokenized_question
    

class CombinedDataset(Dataset):
    def __init__(self, apply_transform=False, img_size=336, use_llm=False, embed_path=None):

        # chunk for original COCO dataloader

        f2 = open("/your-path/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json", "r")
        annotations = json.load(f2)
        f2.close()
        self.sample_list = []
        for sample in annotations:
            if 'image' in sample:
                self.sample_list.append(sample)
        self.num_of_pretraining_data = len(self.sample_list)

        f1 = open("/your-path/LLaVA-Instruct-150K/llava_v1_5_mix665k.json", "r")
        annotations_2 = json.load(f1)
        f1.close()
        for sample in annotations_2:
            if 'image' in sample:
                self.sample_list.append(sample)

        self.train_transform = T.Compose([
                        T.Resize((img_size, img_size)),
                        T.ToTensor(),
                        # T.RandomHorizontalFlip(0.5),
                        T.ColorJitter(brightness=.2, hue=.1),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.val_transform = T.Compose([
                        T.Resize((img_size, img_size)),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
        self.apply_transform = apply_transform

        # chunk for noun phrase extraction -->> creating a prompt from template
        self.template = [
            'a picture of {}.',
            'itap of {}.',
            'a photograph of {}.',
            'this picture contains {}.',
            'a good photo of {}.'
        ]
        self.nlptk = spacy.load("en_core_web_sm")

        # chunk for tokenization
        self.open_clip_tokenizer = oc.get_tokenizer('ViT-L-14-336')
        self.use_llm = use_llm
        self.embed = None
        # the first is template, and the second is full caption
        if embed_path != None:
            self.embed = []
            for p in embed_path:
                embed = np.load(p)
                self.embed.append(embed)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        # chunk for original COCO dataloader
        sample = self.sample_list[index]
        if index < self.num_of_pretraining_data: # from pretraining stage
            img_path = os.path.join("/your-path/LLaVA-Pretrain", "images", sample["image"])
            caption = sample["conversations"][1]['value'] # a python string
        else:
            img_path = os.path.join("/your-path/datamix665k/", sample["image"])
            i = random.randint(0, len(sample["conversations"]) // 2 - 1)
            caption = sample["conversations"][i*2+1]['value'] # a python string
        
        image = Image.open(img_path).convert('RGB')
        if self.apply_transform == True:
            image = self.train_transform(image) # tensor of shape = [3, H, W]
        else:
            image = self.val_transform(image)        
        # use original caption 50% of the time
        nounphrase_or_full_caption = random.choice([0,1])
        # chunk for noun phrase extraction -->> creating a prompt from template
        processed_text = self.nlptk(caption) 
        all_noun_phrases = [chunk.text.lower() for chunk in processed_text.noun_chunks]
        random_template = random.choice(self.template)
        if self.use_llm:
            if index < self.num_of_pretraining_data: 
                if len(all_noun_phrases) != 0 and nounphrase_or_full_caption == 0:
                    tokenized_phrase = self.embed[0][index]
                else:
                    tokenized_phrase = self.embed[1][index]
            else:
                tokenized_phrase = self.embed[2][index-self.num_of_pretraining_data]
        else:
            if len(all_noun_phrases) != 0 and nounphrase_or_full_caption == 0:
                random_noun_phrase = random.choice(all_noun_phrases)
                single_noun_phrase_per_img = random_template.format(random_noun_phrase)
            else:
                single_noun_phrase_per_img = caption
            
            tokenized_phrase = self.open_clip_tokenizer(single_noun_phrase_per_img).squeeze()
        return image, tokenized_phrase
