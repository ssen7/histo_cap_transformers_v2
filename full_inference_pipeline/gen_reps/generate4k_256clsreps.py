import random
import openslide
import h5py
import pandas as pd
import numpy as np

from .hipt_4k import HIPT_4K
from .hipt_model_utils import get_vit256, get_vit4k, eval_transforms
from .hipt_heatmap_utils import *

import torch
import os
import time
# torch.cuda.get_device_name(0),torch.cuda.get_device_name(1)

# df_path='/home/ss4yd/vision_transformer/captioning_vision_transformer/data_files/prepared_prelim_data_gtex4k_left.csv'
# df_path='/home/ss4yd/vision_transformer/captioning_vision_transformer/data_files/left_k_patches.csv'
df_path='/home/ss4yd/nlp/get_female_reps.csv'
df=pd.read_csv(df_path)

# notdonelist=pd.read_pickle('/home/ss4yd/vision_transformer/captioning_vision_transformer/data_files/non_pids.pickle')
# df=df[df.pid.isin(notdonelist)].reset_index(drop=True)
# print(df.shape)

def get_model():
    pretrained_weights256 = '/home/ss4yd/vision_transformer/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth'
    pretrained_weights4k = '/home/ss4yd/vision_transformer/HIPT/HIPT_4K/Checkpoints/vit4k_xs_dino.pth'

    if torch.cuda.is_available():
        device256 = torch.device('cuda:0')
        device4k = torch.device('cuda:1')
    else:
        device256 = torch.device('cpu')
        device4k = torch.device('cpu')


    ### ViT_256 + ViT_4K loaded independently (used for Attention Heatmaps)
    # model256 = get_vit256(pretrained_weights=pretrained_weights256, device=device256)
    # model4k = get_vit4k(pretrained_weights=pretrained_weights4k, device=device4k)

    ### ViT_256 + ViT_4K loaded into HIPT_4K API
    model = HIPT_4K(pretrained_weights256, pretrained_weights4k, device256, device4k)
    model.eval()
    return model

def get_reps(model, slide, coords, size=4096):
    x,y=coords
    region=slide.read_region((x,y),0,(size,size)).convert('RGB')
    x = eval_transforms()(region).unsqueeze(dim=0)
    with torch.no_grad():
        out = torch.tensor(model.forward_asset_dict(x)['features_cls256'])
    return out.cpu()

save_path='/scratch/ss4yd/hipt4k_256cls_reps_new/'
os.makedirs(save_path, exist_ok=True)

# patch_dict={key:value for key,value in zip(df['pid'], df['patch_path'])}

def generate4kreps_onewsi(wsi_path, save_dir):
    
    ### Start Rep Timer
    start_time = time.time()

    patch_dir= os.path.join(save_dir, 'patches')
    rep_save_dir = os.path.join(save_dir, 'reps')
	
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(rep_save_dir, exist_ok=True)
	
    slide_id = wsi_path.split('/')[-1].split('.')[0]
    if os.path.exists(os.path.join(save_dir, 'reps', f'{slide_id}.pt')):
        print("Representations already exist")
        return

    model=get_model()
    
    patch_path = os.path.join(patch_dir,f'{slide_id}.h5')
    
    print('SVS ID: '+f'{slide_id}')
    patch_rep_list=[]
    coords = h5py.File(patch_path, 'r')['coords']
    try:
        slide=openslide.open_slide(wsi_path)
    except:
        print('Slide PID: '+f'{slide_id}'+' skipped')

    print('Number of patches: '+f'{len(coords)}')
    for coord in coords:
        patch_rep_list.append(get_reps(model,slide,coord))
    
    tensor=torch.stack(patch_rep_list)
    torch.save(tensor,os.path.join(rep_save_dir, f'{slide_id}.pt'))
    print('Finished saving tensor..')
    rep_time_elapsed = time.time() - start_time
    print("generating reps took {} seconds".format(rep_time_elapsed))