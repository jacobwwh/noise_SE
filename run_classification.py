import argparse
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

from torch import autocast  #for fp16 (new version instead of apex)
from torch.cuda.amp import GradScaler

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
                          
from utils.poj_utils import ClassificationDataset,generate_data
from utils.codenet_utils import read_codenetdata
from model.bert import bert_classifier_self,lstm_classifier

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument("--metaset", type=boolean_string, default=False, help="use metaset for reweighting")
parser.add_argument("--dataset", type=str, default='poj')
parser.add_argument('--meta_net_hidden_size', type=int, default=100)
parser.add_argument('--meta_net_num_layers', type=int, default=1)
parser.add_argument('--meta_lr', type=float, default=1e-6)
parser.add_argument('--meta_weight_decay', type=float, default=0.)
parser.add_argument('--meta_interval', type=int, default=1)

parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--dampening', type=float, default=0.)
parser.add_argument('--nesterov', type=bool, default=False)
parser.add_argument('--weight_decay', type=float, default=5e-4)

parser.add_argument("--model_type", default="codebert", type=str, help="The model architecture to be fine-tuned.")
parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
args=parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert args.dataset in ['poj','java250','python800']
assert args.model_type in ['codebert','graphcodebert','codet5','unixcoder']
if args.dataset=='poj':
    num_classes=104 #poj
elif args.dataset=='java250':
    num_classes=250 #codenet java250
elif args.dataset=='python800':
    num_classes=800 #codenet python800

if args.model_type=='codebert':
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    encoder_config= RobertaConfig.from_pretrained("microsoft/codebert-base")
    encoder_config.num_labels=num_classes
    model_encoder = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base",config=encoder_config)
    #model_encoder = RobertaForSequenceClassification._from_config(encoder_config) #no pre-trained weights
elif args.model_type=='graphcodebert':
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
    encoder_config= RobertaConfig.from_pretrained("microsoft/graphcodebert-base")
    encoder_config.num_labels=num_classes
    model_encoder = RobertaForSequenceClassification.from_pretrained("microsoft/graphcodebert-base",config=encoder_config)
elif args.model_type=='unixcoder':
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
    encoder_config= RobertaConfig.from_pretrained("microsoft/unixcoder-base")
    encoder_config.num_labels=num_classes
    model_encoder = RobertaForSequenceClassification.from_pretrained("microsoft/unixcoder-base",config=encoder_config)
