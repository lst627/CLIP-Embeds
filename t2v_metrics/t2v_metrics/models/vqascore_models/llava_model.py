from typing import List
import torch
import copy
import re

from .conversation import conv_templates
from .vqa_model import VQAScoreModel
from .mm_utils import expand2square, load_pretrained_model, tokenizer_image_token
from ...constants import HF_CACHE_DIR, CONTEXT_LEN, SYSTEM_MSG, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from ...constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from .llava.model import LlavaLlamaForCausalLM, ModelArguments, LlavaLlama3ForCausalLM, Llama3ModelArguments, LlavaPhiForCausalLM, PhiModelArguments
import transformers

default_question_template = 'Does this figure show "{}"? Please answer yes or no.'
default_answer_template = "Yes"

def format_question(question, conversation_style='chat'):
    if conversation_style == 'plain': # for 1st stage model
        question = DEFAULT_IMAGE_TOKEN + question
    elif conversation_style == 'chat': # for 2nd stage model
        question = SYSTEM_MSG + " USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
    elif conversation_style == 'phi3_instruct' or conversation_style == 'llama3': # for 2nd stage model
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        qs = question
        if IMAGE_PLACEHOLDER in qs:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[conversation_style].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        question = prompt
    # elif conversation_style == 'chat_no_system': # for 2nd stage model
    #     question = "USER: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " ASSISTANT: "
    # elif conversation_style == 'chat_ood_system': # for 2nd stage model
    #     question = SYSTEM_MSG + " HUMAN: " + DEFAULT_IMAGE_TOKEN + "\n" + question + " GPT: "
    # elif conversation_style == 'chat_swap_image_question': # for 2nd stage model
    #     question = SYSTEM_MSG + " HUMAN: " + question + "\n" + DEFAULT_IMAGE_TOKEN + " GPT: "
    else:
        raise NotImplementedError()
    return question

def format_answer(answer, conversation_style='chat'):
    if conversation_style == 'plain': # for 1st stage model
        answer = answer + "\n"
    elif conversation_style == 'chat': # for 2nd stage model
        answer = answer + "</s>"
    elif conversation_style == 'phi3_instruct':
        answer = answer + "<|end|>"
    elif conversation_style == 'llama3':
        answer = answer + "<|eot_id|>"
    # elif conversation_style in ['chat_no_system', 'chat_ood_system', 'chat_swap_image_question']: # for 2nd stage model
    #     answer = answer + "</s>"
    else:
        raise NotImplementedError()
    return answer

LLAVA_LLAMA_MODELS = {
    'llava-phi-3': {
        'tokenizer' : {
            'model_max_length': 2048,
            'padding_side': 'right',
            'path': 'MBZUAI/LLaVA-Phi-3-mini-4k-instruct', 
        },
        'model': {
            'path': 'MBZUAI/LLaVA-Phi-3-mini-4k-instruct',
            'conversation': 'phi3_instruct',
            'image_aspect_ratio': 'pad',
        },
    },
    'llava-llama-3': {
        'tokenizer' : {
            'model_max_length': 4096,
            'padding_side': 'right',
            'path': 'MBZUAI/LLaVA-Meta-Llama-3-8B-Instruct',
        },
        'model': {
            'path': 'MBZUAI/LLaVA-Meta-Llama-3-8B-Instruct',
            'conversation': 'llama3',
            'image_aspect_ratio': 'pad',
        },
    }
}

LLAVA_MODELS = {
    # llava-v1.6
    # 'llava-v1.6-13b': {
    #     'tokenizer' : {
    #         'path': 'liuhaotian/llava-v1.6-vicuna-13b',
    #     },
    #     'model': {
    #         'path': 'liuhaotian/llava-v1.6-vicuna-13b',
    #         'conversation': 'chat',
    #         'image_aspect_ratio': 'pad',
    #     },
    # },
    # We recommend using 'llava-v1.5-13b' for maximal performance.
    # If you want to use a smaller model, we recommend using 'llava-v1.5-7b'.
    'llava-v1.5-13b': {
        'tokenizer' : {
            'path': 'liuhaotian/llava-v1.5-13b',
        },
        'model': {
            'path': 'liuhaotian/llava-v1.5-13b',
            'conversation': 'chat',
            'image_aspect_ratio': 'pad',
        },
    },
    'llava-v1.5-7b': {
        'tokenizer' : {
            'path': 'liuhaotian/llava-v1.5-7b',
        },
        'model': {
            'path': 'liuhaotian/llava-v1.5-7b',
            'conversation': 'chat',
            'image_aspect_ratio': 'pad',
        },
    },
    # The following models are suboptimal, but are included for completeness.
    # 'llava-v1.5-13b-no-system': {
    #     'tokenizer' : {
    #         'path': 'liuhaotian/llava-v1.5-13b',
    #     },
    #     'model': {
    #         'path': 'liuhaotian/llava-v1.5-13b',
    #         'conversation': 'chat_no_system',
    #         'image_aspect_ratio': 'pad',
    #     },
    # },
    # 'llava-v1.5-13b-ood-system': {
    #     'tokenizer' : {
    #         'path': 'liuhaotian/llava-v1.5-13b',
    #     },
    #     'model': {
    #         'path': 'liuhaotian/llava-v1.5-13b',
    #         'conversation': 'chat_ood_system',
    #         'image_aspect_ratio': 'pad',
    #     },
    # },
    # 'llava-v1.5-13b-swap': {
    #     'tokenizer' : {
    #         'path': 'liuhaotian/llava-v1.5-13b',
    #     },
    #     'model': {
    #         'path': 'liuhaotian/llava-v1.5-13b',
    #         'conversation': 'chat_swap_image_question',
    #         'image_aspect_ratio': 'pad',
    #     },
    # },
    # 'llava-v1.5-13b-stage-1': {
    #     'tokenizer' : {
    #         'path': 'lmsys/vicuna-13b-v1.5',
    #         'model_max_length': CONTEXT_LEN,
    #         'padding_side': 'right',
    #     },
    #     'model': {
    #         'path': 'lmsys/vicuna-13b-v1.5',
    #         'mmprojector_repo': 'liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5',
    #         'mmprojector_name': 'mm_projector.bin',
    #         'conversation': "plain",
    #         'image_aspect_ratio': 'square',
    #     },
    # },
    # 'llava-v1.5-7b-stage-1': {
    #     'tokenizer' : {
    #         'path': 'lmsys/vicuna-7b-v1.5',
    #         'model_max_length': CONTEXT_LEN,
    #         'padding_side': 'right',
    #     },
    #     'model': {
    #         'path': 'lmsys/vicuna-7b-v1.5',
    #         'mmprojector_repo': 'liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5',
    #         'mmprojector_name': 'mm_projector.bin',
    #         'conversation': "plain",
    #         'image_aspect_ratio': 'square',
    #     },
    # },
    # The following models are built on top of LLaVA-1.5 and integrate well with LLaVA-1.5 codebase
    'sharegpt4v-7b': {
        'tokenizer' : {
            'path': 'Lin-Chen/ShareGPT4V-7B',
        },
        'model': {
            'path': 'Lin-Chen/ShareGPT4V-7B',
            'conversation': 'chat',
            'image_aspect_ratio': 'pad',
        },
    },
    'sharegpt4v-13b': {
        'tokenizer' : {
            'path': 'Lin-Chen/ShareGPT4V-13B',
        },
        'model': {
            'path': 'Lin-Chen/ShareGPT4V-13B',
            'conversation': 'chat',
            'image_aspect_ratio': 'pad',
        },
    },
}

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



class LLaVAModel(VQAScoreModel):
    """A wrapper for the LLaVA-1.5 models"""
    def __init__(self,
                 model_name='llava-v1.5-13b',
                 device='cuda',
                 cache_dir=HF_CACHE_DIR):
        assert model_name in LLAVA_MODELS
        super().__init__(model_name=model_name,
                         device=device,
                         cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        model_args = ModelArguments()
        model_max_length = LLAVA_MODELS[self.model_name]['tokenizer']['model_max_length'] \
            if 'model_max_length' in LLAVA_MODELS[self.model_name]['tokenizer'] else None
        padding_side = LLAVA_MODELS[self.model_name]['tokenizer']['padding_side'] \
            if 'padding_side' in LLAVA_MODELS[self.model_name]['tokenizer'] else None
        mmprojector_repo = LLAVA_MODELS[self.model_name]['model']['mmprojector_repo'] \
            if 'mmprojector_repo' in LLAVA_MODELS[self.model_name]['model'] else None
        mmprojector_name = LLAVA_MODELS[self.model_name]['model']['mmprojector_name'] \
            if 'mmprojector_name' in LLAVA_MODELS[self.model_name]['model'] else None
        
        # default is 'pad' (llava-1.5 says this reduces hallucination)
        # stage-1 models use 'square'
        self.image_aspect_ratio = LLAVA_MODELS[self.model_name]['model']['image_aspect_ratio'] \
            if 'image_aspect_ratio' in LLAVA_MODELS[self.model_name]['model'] else 'pad'
        
        self.conversational_style = LLAVA_MODELS[self.model_name]['model']['conversation']
        
        self.context_len = CONTEXT_LEN
        
        self.tokenizer, self.model, self.image_processor = load_pretrained_model(
            LlavaLlamaForCausalLM,
            model_args,
            model_path=LLAVA_MODELS[self.model_name]['model']['path'],
            tokenizer_path=LLAVA_MODELS[self.model_name]['tokenizer']['path'],
            model_max_length=model_max_length,
            padding_side=padding_side,
            image_aspect_ratio=self.image_aspect_ratio,
            mmprojector_repo=mmprojector_repo,
            mmprojector_name=mmprojector_name,
            device=self.device,
            cache_dir=self.cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

    def load_images(self,
                    image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (after preprocessing) put on self.device
        """
        image = [self.image_loader(x) for x in image]
        if self.image_aspect_ratio == 'pad':
            image = [expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean)) for image in image]
        image = [self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] for image in image]
        assert all(x.shape == image[0].shape for x in image)
        image = torch.stack(image, dim=0).to(self.device)
        return image

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self,
                images: List[str],
                texts: List[str],
                question_template: str=default_question_template,
                answer_template: str=default_answer_template) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts), "Number of images and texts must match"
        # Turn "a photo of a dog" into
        # Q: "Does this figure show "a photo of a dog"? Please answer yes or no."
        # A: "Yes"
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        
        # Formatting for LLaVA-1.5 desired input including system message and image tokens
        questions = [format_question(question, conversation_style=self.conversational_style) for question in questions]
        answers = [format_answer(answer, conversation_style=self.conversational_style) for answer in answers]
        
        images = self.load_images(images)
        
        prompts = [qs + ans for qs, ans in zip(questions, answers)]
        
        input_ids = [tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt') for prompt in prompts]
        labels = copy.deepcopy(input_ids)
        for label, qs in zip(labels, questions):
            tokenized_len = len(tokenizer_image_token(qs, self.tokenizer))
            if qs[-1] == " ":
                tokenized_len -= 1 # because white space
            label[:tokenized_len] = IGNORE_INDEX
    
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
            
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.model.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            None,
            labels,
            images
        )
        
        assert input_ids is None, "input_ids should be None for LLaVA-1.5"
        assert past_key_values is None, "past_key_values should be None for LLaVA-1.5"
        model_input_kwargs = {
            'input_ids': input_ids, # None for LLaVA-1.5
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': inputs_embeds,
            'use_cache': None,
            'output_attentions': None,
            'output_hidden_states': None,
            'return_dict': False,
        }
        
        outputs = self.model.model(
            **model_input_kwargs
        )

        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        shift_labels = shift_labels.to(shift_logits.device)
        lm_prob = torch.zeros(shift_logits.shape[0])
        for k in range(lm_prob.shape[0]):
            lm_prob[k] = (-loss_fct(shift_logits[k], shift_labels[k])).exp()
        return lm_prob

class LLaVALLaMAModel(VQAScoreModel):
    def __init__(self,
                 model_name='llava-phi-3',
                 device='cuda',
                 cache_dir=HF_CACHE_DIR):
        super().__init__(model_name=model_name,
                         device=device,
                         cache_dir=cache_dir)

    def load_model(self):
        """Load the model, tokenizer, image transform
        """
        model_args = PhiModelArguments() if 'phi' in self.model_name else Llama3ModelArguments()
        model_max_length = LLAVA_LLAMA_MODELS[self.model_name]['tokenizer']['model_max_length'] \
            if 'model_max_length' in LLAVA_LLAMA_MODELS[self.model_name]['tokenizer'] else None
        padding_side = LLAVA_LLAMA_MODELS[self.model_name]['tokenizer']['padding_side'] \
            if 'padding_side' in LLAVA_LLAMA_MODELS[self.model_name]['tokenizer'] else None
        mmprojector_repo = LLAVA_LLAMA_MODELS[self.model_name]['model']['mmprojector_repo'] \
            if 'mmprojector_repo' in LLAVA_LLAMA_MODELS[self.model_name]['model'] else None
        mmprojector_name = LLAVA_LLAMA_MODELS[self.model_name]['model']['mmprojector_name'] \
            if 'mmprojector_name' in LLAVA_LLAMA_MODELS[self.model_name]['model'] else None
        
        # default is 'pad' (llava-1.5 says this reduces hallucination)
        # stage-1 models use 'square'
        self.image_aspect_ratio = LLAVA_LLAMA_MODELS[self.model_name]['model']['image_aspect_ratio'] \
            if 'image_aspect_ratio' in LLAVA_LLAMA_MODELS[self.model_name]['model'] else 'pad'
        
        self.conversational_style = LLAVA_LLAMA_MODELS[self.model_name]['model']['conversation']
        
        model_cls = LlavaPhiForCausalLM if 'phi' in self.model_name else LlavaLlamaForCausalLM 
        self.context_len = CONTEXT_LEN
        self.tokenizer, self.model, self.image_processor = load_pretrained_model(
            model_cls,
            model_args,
            model_path=LLAVA_LLAMA_MODELS[self.model_name]['model']['path'],
            tokenizer_path=LLAVA_LLAMA_MODELS[self.model_name]['tokenizer']['path'],
            model_max_length=model_max_length,
            padding_side=padding_side,
            image_aspect_ratio=self.image_aspect_ratio,
            mmprojector_repo=mmprojector_repo,
            mmprojector_name=mmprojector_name,
            device=self.device,
            cache_dir=self.cache_dir
        )
        if 'phi' in self.model_name:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        elif self.tokenizer.pad_token is None:
            print(f"Adding pad token as '<pad>'")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>"),
                tokenizer=self.tokenizer,
                model=self.model,
            )
        # conversation_lib.default_conversation = conversation_lib.conv_templates[self.conversational_style]
        # print(f"Using conversation format: {conversation_lib.default_conversation.version}")


    def load_images(self,
                    image: List[str]) -> torch.Tensor:
        """Load the image(s), and return a tensor (after preprocessing) put on self.device
        """
        image = [self.image_loader(x) for x in image]
        if self.image_aspect_ratio == 'pad':
            image = [expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean)) for image in image]
        image = [self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] for image in image]
        assert all(x.shape == image[0].shape for x in image)
        image = torch.stack(image, dim=0).to(self.device)
        return image

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def forward(self,
                images: List[str],
                texts: List[str],
                question_template: str=default_question_template,
                answer_template: str=default_answer_template) -> torch.Tensor:
        """Forward pass of the model to return n scores for n (image, text) pairs (in PyTorch Tensor)
        """
        assert len(images) == len(texts), "Number of images and texts must match"
        # Turn "a photo of a dog" into
        # Q: "Does this figure show "a photo of a dog"? Please answer yes or no."
        # A: "Yes"
        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        
        # Formatting for LLaVA-1.5 desired input including system message and image tokens
        questions = [format_question(question, conversation_style=self.conversational_style) for question in questions]
        answers = [format_answer(answer, conversation_style=self.conversational_style) for answer in answers]
        
        images = self.load_images(images)
        
        prompts = [qs + ans for qs, ans in zip(questions, answers)]
        
        input_ids = [tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt') for prompt in prompts]
        labels = copy.deepcopy(input_ids)
        for label, qs in zip(labels, questions):
            tokenized_len = len(tokenizer_image_token(qs, self.tokenizer))
            if qs[-1] == " ":
                tokenized_len -= 1 # because white space
            label[:tokenized_len] = IGNORE_INDEX
    
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
            
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        input_ids, attention_mask, labels = input_ids.to(self.device), attention_mask.to(self.device), labels.to(self.device)
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.model.prepare_inputs_labels_for_multimodal(
            input_ids,
            attention_mask,
            None,
            labels,
            images
        )
        
        assert input_ids is None, "input_ids should be None for LLaVA-1.5"
        assert past_key_values is None, "past_key_values should be None for LLaVA-1.5"
        model_input_kwargs = {
            'input_ids': input_ids, # None for LLaVA-1.5
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'inputs_embeds': inputs_embeds,
            'use_cache': None,
            'output_attentions': None,
            'output_hidden_states': None,
            'return_dict': False,
        }
        
        outputs = self.model.model(
            **model_input_kwargs
        )

        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        shift_labels = shift_labels.to(shift_logits.device)
        lm_prob = torch.zeros(shift_logits.shape[0])
        for k in range(lm_prob.shape[0]):
            lm_prob[k] = (-loss_fct(shift_logits[k], shift_labels[k])).exp()
        return lm_prob