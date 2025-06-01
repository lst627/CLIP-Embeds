import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
os.environ['HF_HOME'] = '/your-path/hf_cache/'
sys.path.append('/your-folder-path/clip_llava')
import open_clip.src.open_clip as oc
from transformers import AutoModel, AutoConfig, AutoTokenizer

"""
PACL model
"""

# for reference, not trained
class plain_clip(torch.nn.Module):
    def __init__(self):
        super(plain_clip, self).__init__()

        self.clip_model, _, _ = oc.create_model_and_transforms('ViT-L-14-336', pretrained='openai')

    def forward_visual(self, images):
        visual_cls = self.clip_model.encode_image(images)
        return visual_cls
    
    def forward_text(self, caps):
        text_cls = self.clip_model.encode_text(caps)
        return text_cls # shape = [B, 768]
    
    def forward(self, images, caps): 
        visual_proj = self.forward_visual(images) # [B, 196, 768]
        text_proj = self.forward_text(caps) # [B, 768]
        return F.normalize(visual_proj, dim=-1), F.normalize(text_proj, dim=-1)


class Patch_Projection(torch.nn.Module):
    def __init__(self, in_dim=768, out_dim=512):
        super(Patch_Projection, self).__init__()
        
        self.linear_projection = self.text_projection = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )
        self.non_linear_projection = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.linear_projection(x) + self.non_linear_projection(x)


class open_clip_pacl(torch.nn.Module):
    def __init__(self, base_model = 'ViT-B-16'):
        super(open_clip_pacl, self).__init__()

        if base_model == 'ViT-B-16':
            self.clip_model, _, _ = oc.create_model_and_transforms('ViT-B-16', pretrained='laion2b-s34b-b88K')
            self.clip_model.visual.positional_embedding = self.interpolate_pos_embed(self.clip_model.visual.positional_embedding.detach(), img_size=400)
            self.visual_projection = nn.Sequential(
                nn.LayerNorm(768),
                nn.Dropout(0.1),
                Patch_Projection(),
            )
            self.text_projection = nn.Sequential(
                nn.LayerNorm(512),
                nn.Dropout(0.1),
                nn.Linear(512, 512),
            )
        elif base_model == 'ViT-L-14-336':
            self.clip_model, _, _ = oc.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
            self.visual_projection = nn.Sequential(
                nn.LayerNorm(1024),
                nn.Dropout(0.1),
                Patch_Projection(1024, 768),
            )
            self.text_projection = nn.Sequential(
                nn.LayerNorm(768),
                nn.Dropout(0.1),
                nn.Linear(768, 768),
            )
        elif base_model == 'EVA01-g-14':
            self.clip_model, _, _ = oc.create_model_and_transforms('EVA01-g-14', pretrained='laion400m_s11b_b41k')
            # B, 256, 1408
            self.visual_projection = nn.Sequential(
                nn.LayerNorm(1408),
                nn.Dropout(0.1),
                Patch_Projection(1408, 1024),
            )
            # B, 1024
            self.text_projection = nn.Sequential(
                nn.LayerNorm(1024),
                nn.Dropout(0.1),
                nn.Linear(1024, 1024),
            )
        else:
            raise NotImplementedError
        
        for p in self.clip_model.parameters(): p.requires_grad=False

        # this makes sure that the unnormalized visual patch tokens are returned
        self.clip_model.visual.output_tokens = True
        

    def interpolate_pos_embed(self, pos_embed, img_size):
        cls_pos_embed, patch_pos_embed = pos_embed[0,:], pos_embed[1:,:] # torch.Size([768]) torch.Size([196, 768])
        new_num_patches = int(img_size // 16) # 25 for img_size=400
        new_patch_pos_embed = patch_pos_embed.reshape(1, 196, 768).transpose(1, 2).reshape(1, 768, 14, 14) # torch.Size([1, 768, 14, 14])
        new_patch_pos_embed = torch.nn.functional.interpolate(new_patch_pos_embed, size=(new_num_patches,new_num_patches), mode='bilinear') # torch.Size([1, 768, 25, 25])
        new_patch_pos_embed = new_patch_pos_embed.reshape(1, 768, 625).transpose(1,2).squeeze(0) # torch.Size([625, 768])
        new_pos_embed = torch.cat((cls_pos_embed.unsqueeze(0), new_patch_pos_embed),dim=0) # torch.Size([626, 768])
        return torch.nn.Parameter(new_pos_embed)      
    
    def forward_visual(self, images):
        visual_cls, visual_patches = self.clip_model.encode_image(images)
        return self.visual_projection(visual_patches) # shape = [B, 196, 768]
    
    def forward_text(self, caps):
        text_cls = self.clip_model.encode_text(caps)
        return self.text_projection(text_cls) # shape = [B, 768]
    
    def patch_alignment(self, visual_patch_proj, text_cls_proj): # shapes =  [B, 196, 768], [B, 768]
        # normalize visual patch tokens and then permute
        normalized_visual_patch_proj = F.normalize(visual_patch_proj, dim=-1)
        normalized_visual_patch_proj = normalized_visual_patch_proj.transpose(-2,-1) # shapes =  [B, 768, 196]
        # normalize text cls token and unsqueeze (required for matmul)
        normalized_text_cls_proj = F.normalize(text_cls_proj, dim=-1)
        normalized_text_cls_proj = normalized_text_cls_proj.unsqueeze(1) # shapes =  [B, 1, 768]

        # compute dot product
        patch_activations = normalized_text_cls_proj @ normalized_visual_patch_proj # shapes =  [B, 1, 196]
        patch_activations = patch_activations.squeeze() # shapes =  [B, 196]
        # because of dot product, the range is between -1 (least similar) to +1 (most similar)
        # multiply by 10 and apply sigmoid function. this squashes the range from 0 to 1 for every element (not necessarily sums to 1 like that of a softmax function)
        return F.sigmoid(patch_activations*10)
    
    def forward(self, images, caps): 
        visual_proj = self.forward_visual(images) # [B, 196, 768]
        # print(visual_proj.shape) (1, 576, 768) for ViT-L-14-336 model
        text_proj = self.forward_text(caps) # [B, 768]
        # computed weighted sum
        patch_activations = self.patch_alignment(visual_proj, text_proj) # shapes =  [B, 196]
        # Eval only !!!!!!
        patch_activations = torch.ones_like(patch_activations)
        patch_pooled_visual_projections = torch.sum(visual_proj * patch_activations.unsqueeze(-1), dim=1) # [B, 768]
        # patch_pooled_visual_projections = torch.sum(visual_proj, dim=1) # [B, 768]
        return F.normalize(patch_pooled_visual_projections, dim=-1), F.normalize(text_proj, dim=-1)

def apply_rope(embeddings):
    """
    Apply Rotary Positional Embeddings to the input embeddings.

    Args:
        embeddings (torch.Tensor): Input embeddings of shape (batch_size, seq_length, dim)

    Returns:
        torch.Tensor: Embeddings after applying RoPE
    """
    batch_size, seq_length, dim = embeddings.size()
    assert dim % 2 == 0, "Embedding dimension must be even for RoPE."

    # Calculate the inverse frequency
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    
    # Create position indices
    position_ids = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)  # (seq_length, 1)
    
    # Compute the angles for each position and dimension
    angles = position_ids * inv_freq  # (seq_length, dim/2)
    
    # Compute sin and cos
    sin_angles = torch.sin(angles).unsqueeze(0).to(embeddings.device)  # (1, seq_length, dim/2)
    cos_angles = torch.cos(angles).unsqueeze(0).to(embeddings.device)  # (1, seq_length, dim/2)
    
    # Split embeddings into even and odd parts
    x1 = embeddings[..., 0::2]  # (batch_size, seq_length, dim/2)
    x2 = embeddings[..., 1::2]  # (batch_size, seq_length, dim/2)
    
    # Apply rotation
    x_rotated = torch.cat([x1 * cos_angles - x2 * sin_angles,
                           x1 * sin_angles + x2 * cos_angles], dim=-1)
    
    return x_rotated


class open_clip_pacl_rope(open_clip_pacl):
    def forward(self, images, caps):
        visual_cls, visual_patches = self.clip_model.encode_image(images)
        rotated_visual_patches = apply_rope(visual_patches)
        visual_proj = self.visual_projection(rotated_visual_patches)  # [B, 196, 768]
        text_proj = self.forward_text(caps) # [B, 768]
        # computed weighted sum
        patch_activations = self.patch_alignment(visual_proj, text_proj) # shapes =  [B, 196]
        # Eval only !!!!!!
        patch_activations = torch.ones_like(patch_activations)
        patch_pooled_visual_projections = torch.sum(visual_proj * patch_activations.unsqueeze(-1), dim=1) # [B, 768]
        
        # patch_pooled_visual_projections = torch.sum(visual_proj, dim=1) # [B, 768]
        return F.normalize(patch_pooled_visual_projections, dim=-1), F.normalize(text_proj, dim=-1)

class open_clip_pacl_rope_after(open_clip_pacl):
    def forward(self, images, caps):
        visual_cls, visual_patches = self.clip_model.encode_image(images)
        visual_proj = self.visual_projection(visual_patches)  # [B, 196, 768]
        rotated_visual_proj = apply_rope(visual_proj)
        text_proj = self.forward_text(caps) # [B, 768]
        # computed weighted sum
        patch_activations = self.patch_alignment(rotated_visual_proj, text_proj) # shapes =  [B, 196]
        patch_pooled_visual_projections = torch.sum(visual_proj * patch_activations.unsqueeze(-1), dim=1) # [B, 768]
        # patch_pooled_visual_projections = torch.sum(visual_proj, dim=1) # [B, 768]
        return F.normalize(patch_pooled_visual_projections, dim=-1), F.normalize(text_proj, dim=-1)


class llm2clip_pacl(torch.nn.Module):
    def __init__(self):
        super(llm2clip_pacl, self).__init__()

        model_name_or_path = "microsoft/LLM2CLIP-Openai-L-14-336" # or /path/to/local/LLM2CLIP-Openai-L-14-336
        self.clip_model = AutoModel.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.float16,
            trust_remote_code=True)
        
        for p in self.clip_model.parameters(): p.requires_grad=False

        self.visual_projection = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Dropout(0.1),
            Patch_Projection(in_dim=1024, out_dim=768),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(1280),
            nn.Dropout(0.1),
            nn.Linear(1280, 768),
        )
        
    def forward_visual(self, images):
        with torch.no_grad(), torch.cuda.amp.autocast():
            visual_outputs = self.clip_model.vision_model(images, return_dict=True)
        visual_patches = visual_outputs.last_hidden_state[:, 1:, :]
        # original visual projection is not used here
        # print(visual_patches.shape, type(visual_patches)) # shape = [B, 576, 1024]
        return self.visual_projection(visual_patches) # shape = [B, 576, 768]
    
    def forward_text(self, text_cls):
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_outputs = self.clip_model.get_text_features(text_cls)
        # print(text_outputs.shape, type(text_outputs)) # shape = [B, 1280]
        return self.text_projection(text_outputs.float()) # shape = [B, 768]
    
    def patch_alignment(self, visual_patch_proj, text_cls_proj):

        # normalize visual patch tokens and then permute
        normalized_visual_patch_proj = F.normalize(visual_patch_proj, dim=-1)
        normalized_visual_patch_proj = normalized_visual_patch_proj.transpose(-2,-1) 
        # normalize text cls token and unsqueeze (required for matmul)
        normalized_text_cls_proj = F.normalize(text_cls_proj, dim=-1)
        normalized_text_cls_proj = normalized_text_cls_proj.unsqueeze(1) # shapes =  [B, 1, 768]

        # compute dot product
        patch_activations = normalized_text_cls_proj @ normalized_visual_patch_proj 
        patch_activations = patch_activations.squeeze()
        # because of dot product, the range is between -1 (least similar) to +1 (most similar)
        # multiply by 10 and apply sigmoid function. this squashes the range from 0 to 1 for every element (not necessarily sums to 1 like that of a softmax function)
        return F.sigmoid(patch_activations*10)
    
    def forward(self, images, text_cls): 
        # caps were embedded by llm2vec!
        visual_proj = self.forward_visual(images) 
        text_proj = self.forward_text(text_cls) # [B, 768]
        # computed weighted sum
        patch_activations = self.patch_alignment(visual_proj, text_proj) 
        # Eval only !!!!!!
        patch_activations = torch.ones_like(patch_activations)
        patch_pooled_visual_projections = torch.sum(visual_proj * patch_activations.unsqueeze(-1), dim=1) 
        # patch_pooled_visual_projections = torch.sum(visual_proj, dim=1) 
        return F.normalize(patch_pooled_visual_projections, dim=-1), F.normalize(text_proj, dim=-1)

class llm2clip_pacl_rope(llm2clip_pacl):
    def forward(self, images, caps):
        with torch.no_grad(), torch.cuda.amp.autocast():
            visual_outputs = self.clip_model.vision_model(images, return_dict=True)
        visual_patches = visual_outputs.last_hidden_state[:, 1:, :]
        rotated_visual_patches = apply_rope(visual_patches)
        visual_proj = self.visual_projection(rotated_visual_patches)  
        text_proj = self.forward_text(caps) # [B, 1280]
        # computed weighted sum
        patch_activations = self.patch_alignment(visual_proj, text_proj) 
        # Eval only !!!!!!
        patch_activations = torch.ones_like(patch_activations)
        patch_pooled_visual_projections = torch.sum(visual_proj * patch_activations.unsqueeze(-1), dim=1) # [B, 1280]
        # patch_pooled_visual_projections = torch.sum(visual_proj, dim=1) # [B, 1280]
        return F.normalize(patch_pooled_visual_projections, dim=-1), F.normalize(text_proj, dim=-1)

class llm2clip_pacl_base(torch.nn.Module):
    def __init__(self):
        super(llm2clip_pacl_base, self).__init__()

        self.clip_model, _, _ = oc.create_model_and_transforms('ViT-B-16', pretrained='laion2b-s34b-b88K')
        self.clip_model.visual.positional_embedding = self.interpolate_pos_embed(self.clip_model.visual.positional_embedding.detach(), img_size=400)
        for p in self.clip_model.parameters(): p.requires_grad=False
        
        model_name_or_path = "microsoft/LLM2CLIP-Openai-L-14-336" # or /path/to/local/LLM2CLIP-Openai-L-14-336
        self.llmclip_model = AutoModel.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch.float16,
            trust_remote_code=True)
        
        for p in self.llmclip_model.parameters(): p.requires_grad=False

        # this makes sure that the unnormalized visual patch tokens are returned
        self.clip_model.visual.output_tokens = True
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            Patch_Projection(),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(1280),
            nn.Dropout(0.1),
            nn.Linear(1280, 512),
        )

    def interpolate_pos_embed(self, pos_embed, img_size):
        cls_pos_embed, patch_pos_embed = pos_embed[0,:], pos_embed[1:,:] # torch.Size([768]) torch.Size([196, 768])
        new_num_patches = int(img_size // 16) # 25 for img_size=400
        new_patch_pos_embed = patch_pos_embed.reshape(1, 196, 768).transpose(1, 2).reshape(1, 768, 14, 14) # torch.Size([1, 768, 14, 14])
        new_patch_pos_embed = torch.nn.functional.interpolate(new_patch_pos_embed, size=(new_num_patches,new_num_patches), mode='bilinear') # torch.Size([1, 768, 25, 25])
        new_patch_pos_embed = new_patch_pos_embed.reshape(1, 768, 625).transpose(1,2).squeeze(0) # torch.Size([625, 768])
        new_pos_embed = torch.cat((cls_pos_embed.unsqueeze(0), new_patch_pos_embed),dim=0) # torch.Size([626, 768])
        return torch.nn.Parameter(new_pos_embed)      
    
    def forward_visual(self, images):
        visual_cls, visual_patches = self.clip_model.encode_image(images)
        return self.visual_projection(visual_patches) # shape = [B, 196, 768]
    
    def forward_text(self, text_cls):
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_outputs = self.llmclip_model.get_text_features(text_cls)
        # print(text_outputs.shape, type(text_outputs)) # shape = [B, 1280]
        return self.text_projection(text_outputs.float()) # shape = [B, 768]
    
    def patch_alignment(self, visual_patch_proj, text_cls_proj):

        # normalize visual patch tokens and then permute
        normalized_visual_patch_proj = F.normalize(visual_patch_proj, dim=-1)
        normalized_visual_patch_proj = normalized_visual_patch_proj.transpose(-2,-1) 
        # normalize text cls token and unsqueeze (required for matmul)
        normalized_text_cls_proj = F.normalize(text_cls_proj, dim=-1)
        normalized_text_cls_proj = normalized_text_cls_proj.unsqueeze(1) # shapes =  [B, 1, 768]

        # compute dot product
        patch_activations = normalized_text_cls_proj @ normalized_visual_patch_proj 
        patch_activations = patch_activations.squeeze()
        # because of dot product, the range is between -1 (least similar) to +1 (most similar)
        # multiply by 10 and apply sigmoid function. this squashes the range from 0 to 1 for every element (not necessarily sums to 1 like that of a softmax function)
        return F.sigmoid(patch_activations*10)
    
    def forward(self, images, text_cls): 
        # caps were embedded by llm2vec!
        visual_proj = self.forward_visual(images) 
        text_proj = self.forward_text(text_cls) # [B, 768]
        # computed weighted sum
        patch_activations = self.patch_alignment(visual_proj, text_proj) 
        patch_pooled_visual_projections = torch.sum(visual_proj * patch_activations.unsqueeze(-1), dim=1) 
        # patch_pooled_visual_projections = torch.sum(visual_proj, dim=1) 
        return F.normalize(patch_pooled_visual_projections, dim=-1), F.normalize(text_proj, dim=-1)

class llm2clip_pacl_rope_base(llm2clip_pacl_base):
    def forward(self, images, caps):
        visual_cls, visual_patches = self.clip_model.encode_image(images)
        rotated_visual_patches = apply_rope(visual_patches)
        visual_proj = self.visual_projection(rotated_visual_patches)  # [B, 196, 768]
        text_proj = self.forward_text(caps) # [B, 1280]
        # computed weighted sum
        patch_activations = self.patch_alignment(visual_proj, text_proj) 
        patch_pooled_visual_projections = torch.sum(visual_proj * patch_activations.unsqueeze(-1), dim=1) # [B, 1280]
        # patch_pooled_visual_projections = torch.sum(visual_proj, dim=1) # [B, 1280]
        return F.normalize(patch_pooled_visual_projections, dim=-1), F.normalize(text_proj, dim=-1)


class sparc(torch.nn.Module):
    def __init__(self, sigma=1.0/625, base_model = 'ViT-B-16'):
        super(sparc, self).__init__()

        self.sigma = sigma # usually 1 / number of patch tokens
        if base_model == 'ViT-B-16':
            self.clip_model, _, _ = oc.create_model_and_transforms('ViT-B-16', pretrained='laion2b-s34b-b88K')
            self.clip_model.visual.positional_embedding = self.interpolate_pos_embed(self.clip_model.visual.positional_embedding.detach(), img_size=400)
            self.visual_projection = nn.Sequential(
                nn.LayerNorm(768),
                nn.Dropout(0.1),
                Patch_Projection(),
            )
            self.text_projection = nn.Sequential(
                nn.LayerNorm(512),
                nn.Dropout(0.1),
                nn.Linear(512, 512),
            )
        elif base_model == 'ViT-L-14-336':
            self.clip_model, _, _ = oc.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
            self.visual_projection = nn.Sequential(
                nn.LayerNorm(1024),
                nn.Dropout(0.1),
                Patch_Projection(1024, 768),
            )
            self.text_projection = nn.Sequential(
                nn.LayerNorm(768),
                nn.Dropout(0.1),
                nn.Linear(768, 768),
            )
        else:
            raise NotImplementedError
        for p in self.clip_model.parameters(): p.requires_grad=False

        # this makes sure that the unnormalized visual patch tokens are returned
        self.clip_model.visual.output_tokens = True
        self.clip_model.output_text_tokens = True

    def interpolate_pos_embed(self, pos_embed, img_size):
        cls_pos_embed, patch_pos_embed = pos_embed[0,:], pos_embed[1:,:] # torch.Size([768]) torch.Size([196, 768])
        new_num_patches = int(img_size // 16) # 25 for img_size=400
        new_patch_pos_embed = patch_pos_embed.reshape(1, 196, 768).transpose(1, 2).reshape(1, 768, 14, 14) # torch.Size([1, 768, 14, 14])
        new_patch_pos_embed = torch.nn.functional.interpolate(new_patch_pos_embed, size=(new_num_patches,new_num_patches), mode='bilinear') # torch.Size([1, 768, 25, 25])
        new_patch_pos_embed = new_patch_pos_embed.reshape(1, 768, 625).transpose(1,2).squeeze(0) # torch.Size([625, 768])
        new_pos_embed = torch.cat((cls_pos_embed.unsqueeze(0), new_patch_pos_embed),dim=0) # torch.Size([626, 768])
        return torch.nn.Parameter(new_pos_embed)      
    
    def forward_visual(self, images):
        visual_cls, visual_patches = self.clip_model.encode_image(images)
        return self.visual_projection(visual_patches) # shape = [B, 196, 768]
    
    def forward_text(self, caps):
        text_cls, text_tokens = self.clip_model.encode_text(caps)
        eots = caps.argmax(dim=-1)
        language_mask = torch.arange(text_tokens.size(1), device=caps.device).expand(caps.size(0), -1) <= eots.unsqueeze(1)
        language_mask = language_mask.float()  # Convert boolean mask to float mask (0.0 or 1.0)
        return self.text_projection(text_tokens), language_mask 

    def scoring(self, images, caps, local=False):
        if images.shape[0] == 1 and caps.shape[0] > 1:
            images = images.expand(caps.shape[0], *images.shape[1:])
        assert images.shape[0] == caps.shape[0]
        v_patch_embed, l_token_embed, l_grouped_v_patch_embed, language_mask = self.forward(images, caps)
        global_text_features = F.normalize(torch.mean(l_token_embed, dim=1), dim=-1)
        if not local:
            global_image_features = F.normalize(torch.mean(v_patch_embed, dim=1), dim=-1)
            logits_per_image = (global_image_features @ global_text_features.T) # * self.logit_scale
            return logits_per_image

        sum_local_image_features = F.normalize(torch.mean(l_grouped_v_patch_embed, dim=1), dim=-1)
        logits_per_image = (sum_local_image_features @ global_text_features.T) # * self.logit_scale
        return logits_per_image
    
    def forward(self, images, caps): 
        v_patch_embed = self.forward_visual(images) # [B, 196, 768]
        l_token_embed, language_mask = self.forward_text(caps) # [B, 768]
        # print(v_patch_embed.shape, l_token_embed.shape, language_mask.shape)
        # [512, 625, 512], [512, 77, 512], [512, 77]

        # Similarity calculation
        similarity = torch.einsum('btd,bpd->btp', l_token_embed, v_patch_embed)

        # Min-max normalization
        similarity_min = similarity.min(dim=-1, keepdim=True)[0]
        similarity_max = similarity.max(dim=-1, keepdim=True)[0]
        similarity = (similarity - similarity_min) / (similarity_max - similarity_min + 1e-8)

        # Thresholding
        similarity = torch.where(similarity < self.sigma, 0.0, similarity)

        # Alignment-weighting
        v_align_weights = similarity / (similarity.sum(dim=-1, keepdim=True) + 1e-8)
        l_grouped_v_patch_embed = torch.einsum('btp,bpd->btd', v_align_weights, v_patch_embed)

        # L2 Normalization
        l_grouped_v_patch_embed = F.normalize(l_grouped_v_patch_embed, p=2, dim=-1)
        l_token_embed = F.normalize(l_token_embed, p=2, dim=-1)
        
        return v_patch_embed, l_token_embed, l_grouped_v_patch_embed, language_mask

class sparc_rope(sparc):
    def forward_visual(self, images):
        visual_cls, visual_patches = self.clip_model.encode_image(images)
        rotated_visual_patches = apply_rope(visual_patches)
        return self.visual_projection(rotated_visual_patches)

"""
CLIP loss or Image-Text-Contrastive loss
"""
class ClipLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.logit_scale = 1.0/temperature

    def get_ground_truth(self, device, num_logits):
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, image_features, text_features):
        logits_per_image = self.logit_scale * image_features @ text_features.T
        logits_per_text = self.logit_scale * text_features @ image_features.T
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss

class SparcLoss(ClipLoss):
    def __init__(self, temperature):
        super().__init__(temperature)
        self.global_weight = 0.5 
        self.local_weight = 1.0

    def masked_pairwise_contrastive_loss(self, a, b, mask):
        """
        Compute the masked pairwise contrastive loss.
        
        Args:
            a (torch.Tensor): Tensor of shape (batch_size, seq_len, embed_dim).
            b (torch.Tensor): Tensor of shape (batch_size, seq_len, embed_dim).
            mask (torch.Tensor): Mask of shape (batch_size, seq_len) with valid positions as 1 and others as 0.
            inverse_temperature (float): Scaling factor for logits.
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        batch_size, seq_len, _ = a.shape

        # Mask logits to avoid invalid entries
        mask_logits = (1.0 - mask) * (-1e8) # [batch, seq_len]

        # Generate labels (identity matrix, expanded to batch size)
        labels = torch.eye(seq_len, device=a.device).unsqueeze(0).expand(batch_size, -1, -1)
        labels = labels.reshape(batch_size*seq_len, -1)

        # Compute logits (scaled dot product)
        # print(a.shape, b.shape) # [1024, 77, 512], [1024, 77, 512]
        logits = torch.einsum('bmd,bnd->bmn', a, b) * self.logit_scale # [batch, seq_len, seq_len]
        # print(logits.shape, mask_logits.shape) [1024, 77, 77], [1024, 77]
        # Mask out invalid logits
        logits = logits + mask_logits.unsqueeze(1)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits.reshape(batch_size*seq_len, -1), labels, reduction='none')
        mask = mask.reshape(-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def forward(self, v_patch_embed, l_token_embed, l_grouped_v_patch_embed, language_mask):
        device = v_patch_embed.device
        ##### GLOBAL LOSS
        global_image_features = F.normalize(torch.mean(v_patch_embed, dim=1), dim=-1)
        global_text_features = F.normalize(torch.mean(l_token_embed, dim=1), dim=-1)
        logits_per_image, logits_per_text = self.get_logits(global_image_features, global_text_features)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        global_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        ##### LOCAL LOSS 

        # Loss calculation
        loss_vl_local = self.masked_pairwise_contrastive_loss(
            l_grouped_v_patch_embed, l_token_embed, language_mask
        )
        loss_lv_local = self.masked_pairwise_contrastive_loss(
            l_token_embed, l_grouped_v_patch_embed, language_mask
        )

        local_loss = (loss_vl_local + loss_lv_local) / 2
        total_loss = self.global_weight * global_loss + self.local_weight * local_loss
        return total_loss
