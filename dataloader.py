import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision import transforms

import random
# import openslide
import h5py

from PIL import Image
from transformers import AutoTokenizer

class TCGADataset(Dataset):

    def __init__(self, df_path, dtype='train'):
        self.df_path = df_path
        df = pd.read_pickle(self.df_path)
        self.dtype=dtype
        self.df = df[df['dtype']==self.dtype]
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        reps_path=self.df.iloc[idx]['reps_path']
        pixel_values = torch.load(reps_path)
        
        n_imgs = pixel_values.shape[0]
        n_imgs = torch.LongTensor([n_imgs])

        labels=torch.tensor(self.df.iloc[idx]['label'])
        
        if self.dtype=='train':
            return pixel_values, labels
        else:
            # all_captions=torch.LongTensor([idx_tokens])
            return pixel_values, labels
        
class ResnetPlusVitDataset(Dataset):

    def __init__(self, df_path, text_decode_model, dtype='train', th_transform=None, pid_list=None, max_len=128):
        self.df_path = df_path
        df = pd.read_pickle(self.df_path)
        self.dtype=dtype
        if pid_list==None:
            self.df=df[df.dtype==dtype]
        else:
            self.df=df[df.pid.isin(pid_list)]
        self.th_transform = th_transform
        self.max_len = max_len

        self.text_decode_model = text_decode_model
        self.tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        reps_path=self.df.iloc[idx]['reps_path']        
        pixel_values = torch.load(reps_path)
        
        n_imgs = pixel_values.shape[0]
        n_imgs = torch.LongTensor([n_imgs])

        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = 'right'
        # labels, attention_mask = self.tokenizer(self.df.iloc[idx]['notes'],return_tensors='pt', padding='max_length', max_length=64).values()
        encoded_dict = self.tokenizer.encode_plus(
            self.df.iloc[idx]['new_notes'],
            return_tensors='pt',
            add_special_tokens = True,
            max_length = self.max_len,
            padding='max_length',
            return_attention_mask = True,
            )
        
        labels = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']

        if self.th_transform is not None:
            th_img = self.th_transform(th_img)
        
        if self.dtype=='train':
            return pixel_values, labels, attention_mask
        else:
            # all_captions=torch.LongTensor([idx_tokens])
            return pixel_values, labels, attention_mask

class ResnetPlusVitDatasetV2(Dataset):

    def __init__(self, df_path, text_decode_model, dtype='train', th_transform=None, pid_list=None, txt_max_len=128, img_max_len=120):
        self.df_path = df_path
        df = pd.read_pickle(self.df_path)
        self.dtype=dtype
        if pid_list==None:
            self.df=df[df.dtype==dtype]
        else:
            self.df=df[df.pid.isin(pid_list)]
        self.th_transform = th_transform
        self.txt_max_len = txt_max_len
        self.img_max_len = img_max_len

        self.text_decode_model = text_decode_model
        self.tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        reps_path=self.df.iloc[idx]['reps_path']        
        pixel_values = torch.load(reps_path)
        
        n_imgs = pixel_values.shape[0]
        if n_imgs <= self.img_max_len:
            pad_len=self.img_max_len - n_imgs
            pads=torch.zeros((pad_len,pixel_values.shape[1], pixel_values.shape[2]))
            encoder_attention_mask=torch.FloatTensor([1]*n_imgs +[0]*(pad_len))
            pixel_values = torch.vstack([pixel_values, pads])
        else:
            pixel_values=pixel_values[:self.img_max_len,:,:]
            encoder_attention_mask=torch.FloatTensor([1]*self.img_max_len)

        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = 'right'
        # labels, attention_mask = self.tokenizer(self.df.iloc[idx]['notes'],return_tensors='pt', padding='max_length', max_length=64).values()
        encoded_dict = self.tokenizer.encode_plus(
            self.df.iloc[idx]['new_notes'],
            return_tensors='pt',
            add_special_tokens = True,
            max_length = self.txt_max_len,
            padding='max_length',
            return_attention_mask = True,
            )
        
        labels = encoded_dict['input_ids'][0]
        attention_mask = encoded_dict['attention_mask'][0]

        if self.th_transform is not None:
            th_img = self.th_transform(th_img)
        
        if self.dtype=='train':
            return pixel_values, labels, attention_mask, encoder_attention_mask
        else:
            # all_captions=torch.LongTensor([idx_tokens])
            return pixel_values, labels, attention_mask, encoder_attention_mask


def train_collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    pixel_values, labels, attention_mask, n_patches = zip(*data)
    
    pixel_values = torch.vstack(pixel_values)
    labels = torch.vstack(labels)
    attention_mask = torch.vstack(attention_mask)
    n_patches = torch.vstack([x.unsqueeze(0) for x in n_patches])

    return pixel_values, labels, attention_mask, n_patches

class ResnetPlusVitDatasetV3(Dataset):

    def __init__(self, df_path, text_decode_model, dtype='train', th_transform=None, pid_list=None, txt_max_len=128, img_max_len=128):
        self.df_path = df_path
        df = pd.read_pickle(self.df_path)
        self.dtype=dtype
        if pid_list==None:
            self.df=df[df.dtype==dtype]
        else:
            self.df=df[df.pid.isin(pid_list)]
        self.th_transform = th_transform
        self.txt_max_len = txt_max_len
        self.img_max_len = img_max_len

        self.text_decode_model = text_decode_model
        self.tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        reps256_path=self.df.iloc[idx]['reps_path']
        reps4k_path=self.df.iloc[idx]['reps4kpath']
        
        x256 = torch.load(reps256_path)
        x256mean = x256.mean(dim=1)
        x4k = torch.load(reps4k_path)

        pixel_values = torch.cat([x256mean, x4k], dim=1)
        
        n_imgs = pixel_values.shape[0]
        if n_imgs <= self.img_max_len:
            pad_len=self.img_max_len - n_imgs
            pads=torch.zeros((pad_len,pixel_values.shape[1]))
            encoder_attention_mask=torch.FloatTensor([1]*(n_imgs) +[0]*(pad_len)) # +1 for attention weighted token
            pixel_values = torch.vstack([pixel_values, pads])
        else:
            pixel_values=pixel_values[:self.img_max_len,:]
            encoder_attention_mask=torch.FloatTensor([1]*(self.img_max_len)) # +1 for attention weighted token

        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = 'right'
        # labels, attention_mask = self.tokenizer(self.df.iloc[idx]['notes'],return_tensors='pt', padding='max_length', max_length=64).values()
        encoded_dict = self.tokenizer.encode_plus(
            self.df.iloc[idx]['new_notes'],
            return_tensors='pt',
            add_special_tokens = True,
            max_length = self.txt_max_len,
            padding='max_length',
            return_attention_mask = True,
            )
        
        labels = encoded_dict['input_ids'][0]
        attention_mask = encoded_dict['attention_mask'][0]

        if self.th_transform is not None:
            th_img = self.th_transform(th_img)
        
        if (self.dtype=='train') | (self.dtype=='val'):
            return pixel_values, labels, attention_mask, encoder_attention_mask
        else:
            # all_captions=torch.LongTensor([idx_tokens])
            return pixel_values, labels, attention_mask, encoder_attention_mask