import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader
import os
os.environ['HF_HOME'] = '/your-path/hf_cache/'
from data.image_caption_data import CombinedDataset
from data.utils import cosine_lr
from model.pacl import (
    open_clip_pacl, ClipLoss, open_clip_pacl_rope, open_clip_pacl_rope_after, plain_clip,
    llm2clip_pacl, llm2clip_pacl_rope, llm2clip_pacl_base, llm2clip_pacl_rope_base)
import time
import numpy as np


def train_one_epoch(train_data_loader, model, optimizer, loss_fn, device, scheduler, num_of_steps):
    
    epoch_loss = []
   
    model.train()

    begin = time.time()
    ###Iterating over data loader
    for i, (images, caps) in enumerate(train_data_loader):
        if scheduler != None: scheduler(i + num_of_steps)

        #Loading data and labels to device
        images = images.to(device)
        caps = caps.to(device)

        #Reseting Gradients
        optimizer.zero_grad()
        #Forward
        visual_features, text_features = model(images, caps)
        #Calculating Loss
        _loss = loss_fn(visual_features, text_features)
        epoch_loss.append(_loss.item())      
        #Backward
        _loss.backward()
        optimizer.step()
    
        if i%10 == 0: 
            print("train_loss = ",_loss.item())
            elapsed_time = time.time() - begin
            estimated_total_time = elapsed_time * (len(train_data_loader) - i - 1) / (i + 1)
            print(f"Elapsed time: {elapsed_time:.2f}s, Estimated remaining time: {estimated_total_time:.2f}s")

    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss

def val_one_epoch(val_data_loader, model, loss_fn, device):
    
    ### Local Parameters
    epoch_loss = []

    model.eval()

    with torch.no_grad():
        ###Iterating over data loader
        for i, (images, caps) in enumerate(val_data_loader):
        
            #Loading data and labels to device
            images = images.to(device)
            caps = caps.to(device)

            #Forward
            visual_features, text_features = model(images, caps)
            #Calculating Loss
            _loss = loss_fn(visual_features, text_features)
            epoch_loss.append(_loss.item())

    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss


def train_clip(batch_size, epochs):
    """
    DataLoader
    """
    # Create the dataset and dataloader
    model_name = 'pacl_rope_llm_all'
    if "llm" in model_name:
        train_dataset = CombinedDataset(apply_transform=True, img_size=336, use_llm=True, embed_path=["/your-path/single_embed.npy", "/your-path/full_embed.npy", "/your-path/datamix_full_embed.npy"])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    else:
        train_dataset = CombinedDataset(apply_transform=True, img_size=336) 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    """
    Model and Loss
    """
    model = llm2clip_pacl_rope()
    device = torch.device("cuda:0")
    model = nn.DataParallel(model,device_ids=[0,1,2,3,4,5,6,7])
    model.to(device)
    print("\n\n\n\n\t Model Loaded")
    print("\t Total Params = ",sum(p.numel() for p in model.parameters()))
    print("\t Trainable Params = ",sum(p.numel() for p in model.parameters() if p.requires_grad))

    """
    Train
    """
    num_batches = math.ceil(len(train_dataset) // batch_size)
    num_of_steps = 0
    loss_fn = ClipLoss(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler = None

    # for plain clip training, for reference
    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=5e-6,
    #     betas=(0.9, 0.98),
    #     eps=1e-6,
    # )
    # scheduler = cosine_lr(optimizer, base_lr=5e-6, warmup_length=50, steps=epochs * num_batches)

    print("\n\t Started Training\n")

    for epoch in range(epochs):

        begin = time.time()

        ###Training
        loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device, scheduler, num_of_steps)
        num_of_steps += num_batches

        print('\n\n\t Epoch....', epoch + 1)
        print("\t Training loss ......",round(loss,4))
        print('\t Time per epoch (in mins) = ', round((time.time()-begin)/60,2),'\n\n')

    torch.save(model.state_dict(), model_name+'.pth')

if __name__=="__main__":
    train_clip(batch_size=4096, epochs=10)