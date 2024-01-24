# Debugging Neural Networks: https://benjamin-computer.medium.com/debugging-neural-networks-6fa65742efd
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torchmetrics
from watermark import watermark

import os
import time

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .dataloader import *
from .utils import *
from .model_vit_bert import ViTConfigCustom, ViTModelCustom, CustomVEDConfig, CustomVisionEncoderDecoder

# from eval_utils import custom_evaluate, custom_evaluate_only_resnet, custom_evaluate_only_resnet_plus, custom_evaluate_only_vit
from nltk.translate.bleu_score import corpus_bleu

from pytorch_lightning import seed_everything
import argparse
import pdb

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from pytorch_lightning.utilities import grad_norm

import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

class LightningModel(L.LightningModule):
    def __init__(self, model, tokenizer, model_lr=1e-3, max_gen_len=128):
        super().__init__()

        self.save_hyperparameters(ignore=["model","tokenizer"])

        self.model_lr = model_lr  # learning rate for encoder if fine-tuning
        self.tokenizer=tokenizer
        self.model = model
        self.max_gen_len=128
        
        self.val_hypotheses=[]
        self.val_references=[]
        
        self.test_hypotheses=[]
        self.test_hypotheses_w_sptokens=[]
        self.test_references=[]
        
        self.val_bleu = torchmetrics.BLEUScore()
        self.test_bleu = torchmetrics.BLEUScore()
    

    def training_step(self, batch, batch_idx):
        pixel_values, labels, attention_mask = batch
        labels, attention_mask = labels.squeeze(0), attention_mask.squeeze(0)
        
        loss = self.model(pixel_values=pixel_values, labels=labels, decoder_attention_mask=attention_mask).loss
        
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        
        return loss
    

    def validation_step(self, batch, batch_idx):
        pixel_values, labels, attention_mask = batch
        labels, attention_mask = labels.squeeze(0), attention_mask.squeeze(0)
        
        loss = self.model(pixel_values=pixel_values, labels=labels, decoder_attention_mask=attention_mask).loss
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        pixel_values, labels, attention_mask = batch
        labels, attention_mask = labels.squeeze(0), attention_mask.squeeze(0)
        caption_ids = self.model.generate(pixel_values, max_length=self.max_gen_len)
        # print(caption_ids.shape)
        # print(labels.shape)
        caption = self.tokenizer.decode(caption_ids[0], skip_special_tokens=True)
        caption_with_special_tokens = self.tokenizer.decode(caption_ids[0])

        self.test_hypotheses += [caption]
        self.test_hypotheses_w_sptokens += [caption_with_special_tokens]
        self.test_references += [[self.tokenizer.decode(labels[0], skip_special_tokens=True)]]

    def on_test_epoch_end(self):
        references, hypotheses = self.test_references, self.test_hypotheses
        hypwsptokens = self.test_hypotheses_w_sptokens
        # print("Hypotheses: {}".format(hypotheses))
        # print("References: {}".format(references))

        # save preds of test to dataframe
        df=pd.DataFrame()
        df['preds']=hypotheses
        df['target']=[item for sublist in references for item in sublist]
        df['pred_w_sp_tokens']=hypwsptokens
        df.to_csv(os.path.join(self.logger.log_dir, "saved_test.csv"), index=False)
        
        self.test_bleu(hypotheses,references)

        self.log("test_bleu", self.test_bleu, prog_bar=True)
        
        self.test_hypotheses=[]
        self.test_references=[]

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.model_lr)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=8),
            "monitor": "val_loss",
            "frequency": 1,
            "interval":"epoch"
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
            },
        }