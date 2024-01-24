import random
import openslide
import h5py
import pandas as pd
import numpy as np

from .hipt_256 import HIPT_256
from .hipt_model_utils import eval_transforms
from .hipt_heatmap_utils import *

from vision_transformer4k import vit4k_xs

import torch
import os
import time
# torch.cuda.get_device_name(0),torch.cuda.get_device_name(1)

# df_path='/home/ss4yd/vision_transformer/captioning_vision_transformer/data_files/prepared_prelim_data_gtex4k_left.csv'
# df_path='/home/ss4yd/vision_transformer/captioning_vision_transformer/data_files/left_k_patches.csv'
# df_path='/home/ss4yd/nlp/get_female_reps.csv'
# df=pd.read_csv(df_path)

# notdonelist=pd.read_pickle('/home/ss4yd/vision_transformer/captioning_vision_transformer/data_files/non_pids.pickle')
# df=df[df.pid.isin(notdonelist)].reset_index(drop=True)
# print(df.shape)

def get_model256():
    pretrained_weights256 = '/home/ss4yd/vision_transformer/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth'
    pretrained_weights4k = '/home/ss4yd/vision_transformer/HIPT/HIPT_4K/Checkpoints/vit4k_xs_dino.pth'

    if torch.cuda.is_available():
        device256 = torch.device('cuda:0')
    else:
        device256 = torch.device('cpu')


    ### ViT_256 + ViT_4K loaded independently (used for Attention Heatmaps)
    # model256 = get_vit256(pretrained_weights=pretrained_weights256, device=device256)
    # model4k = get_vit4k(pretrained_weights=pretrained_weights4k, device=device4k)

    ### ViT_256 + ViT_4K loaded into HIPT_4K API
    model = HIPT_256(pretrained_weights256, device256)
    model.eval()
    return model

def get_model4k():
    pretrained_weights256 = '/home/ss4yd/vision_transformer/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth'
    pretrained_weights4k = '/home/ss4yd/vision_transformer/HIPT/HIPT_4K/Checkpoints/vit4k_xs_dino.pth'

    if torch.cuda.is_available():
        device4k = torch.device('cuda:0')
    else:
        device4k = torch.device('cpu')


    local_vit = vit4k_xs()

    print("Loading Pretrained Local VIT model...",)
    state_dict = torch.load(f'{pretrained_weights4k}', map_location='cpu')['teacher']
    state_dict = {k.replace('module.', ""): v for k, v in state_dict.items()}
    state_dict = {k.replace('backbone.', ""): v for k, v in state_dict.items()}
    missing_keys, unexpected_keys = local_vit.load_state_dict(state_dict, strict=False)
    print("Done!")
    local_vit=local_vit.to(device4k)
    
    local_vit.eval()
    return local_vit

def get_reps(model, slide, coords, size=4096):
    x,y=coords
    region=slide.read_region((x,y),0,(size,size)).convert('RGB')
    x = eval_transforms()(region).unsqueeze(dim=0)
    with torch.no_grad():
        out = torch.tensor(model.forward_asset_dict(x)['features_cls256'])
    return out.cpu()

# patch_dict={key:value for key,value in zip(df['pid'], df['patch_path'])}

def generate256reps_onewsi(wsi_path, save_dir):
    
    ### Start Rep Timer
    start_time = time.time()

    patch_dir= os.path.join(save_dir, 'patches')
    rep_save_dir = os.path.join(save_dir, 'reps/ft256')
	
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(rep_save_dir, exist_ok=True)
	
    slide_id = wsi_path.split('/')[-1].split('.')[0]
    if os.path.exists(os.path.join(save_dir, 'reps/ft256', f'{slide_id}.pt')):
        print("Representations already exist")
        return

    model=get_model256()
    
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

def generate4kreps_onewsi(wsi_path, save_dir):
    start_time = time.time()

    slide_id = wsi_path.split('/')[-1].split('.')[0]

    patch_dir= os.path.join(save_dir, 'patches')
    rep_save_dir = os.path.join(save_dir, 'reps/ft4k')
	
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(rep_save_dir, exist_ok=True)
	
    slide_id = wsi_path.split('/')[-1].split('.')[0]
    if os.path.exists(os.path.join(save_dir, 'reps/ft4k', f'{slide_id}.pt')):
        print("Representations already exist")
        return

    model=get_model4k()

    reps256path = os.path.join(save_dir, 'reps/ft256', f'{slide_id}.pt')

    if torch.cuda.is_available():
        device4k = torch.device('cuda:0')
    else:
        device4k = torch.device('cpu')

    pixel_values = torch.load(reps256path).to(device4k)
    hipt4kreps=model(pixel_values.unfold(1, 16, 16).transpose(1,2)).cpu()
    torch.save(hipt4kreps,os.path.join(rep_save_dir, f'{slide_id}.pt'))
    
    print('Finished saving tensor..')
    rep_time_elapsed = time.time() - start_time
    print("generating reps took {} seconds".format(rep_time_elapsed))

def generate4kreps_allwsi(df, save_path):
    
    ### Start Rep Timer
    # start_time = time.time()
    done_pids = [x.split('.')[0] for x in os.listdir(save_path)]
    
    model=get_model4k()

    for index, row in df.iterrows():
        patch_path=row['patch_path']
        svs_path=row['svs_path']
        pid=row['pid']
        if pid in done_pids:
            print('\nSVS ID: '+f'{pid}'+' skipped')
            continue
        print('\nSVS ID: '+f'{pid}')
        patch_rep_list=[]
        coords = h5py.File(patch_path, 'r')['coords']
        if len(coords)==0:
            continue
        try:
            slide=openslide.open_slide(svs_path)
        except:
            print('Slide PID: '+f'{pid}'+' skipped')
            continue
        print('Number of patches: '+f'{len(coords)}')
        for coord in coords:
            patch_rep_list.append(get_reps(model,slide,coord))
        
        tensor=torch.stack(patch_rep_list)
        torch.save(tensor,os.path.join(save_path, f'{pid}.pt'))
        print('Finished saving tensor..')
        print('Progress: '+ f'{(index*100)/len(df)}%')


if __name__=='__main__':
    
    save_path='/project/GutIntelligenceLab/ss4yd/gdc_data/hipt256reps/'
    os.makedirs(save_path, exist_ok=True)
    df=pd.read_pickle('/home/ss4yd/nlp/full_inference_pipeline/gen_reps/df_partition_3.pickle')

    generate4kreps_allwsi(df, save_path)



    