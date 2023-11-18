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
                          
from utils.dataset_utils import ClassificationDataset
from utils.codenet_utils import read_codenetdata
from model.bert import bert_classifier_self,lstm_classifier,bert_and_linear_classifier

from utils.codenet_graph_utils import get_spt_dataset,GraphClassificationDataset
from dgl.dataloading import GraphDataLoader
from model.gnn import GNN_codenet

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='java250')

parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.5)
parser.add_argument("--noise_pattern", default="random", type=str, help="Noise pattern(random/flip/).")

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
parser.add_argument("--epochs", default=50, type=int, help="Training epochs.")
parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
args=parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert args.dataset in ['java250','python800']
assert args.model_type in ['codebert','graphcodebert','codet5','unixcoder','gcn','gin','ggnn','hgt','lstm']
if args.dataset=='java250':
    num_classes=250 #codenet java250
elif args.dataset=='python800':
    num_classes=800 #codenet python800

if args.model_type not in ['gcn','gin','ggnn','hgt']:
    if args.model_type=='codebert' or args.model_type=='lstm':
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

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
        args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    model_encoder.to(device)


if args.model_type in ['gcn','gin','ggnn','hgt']:
    print('use gnn: ',args.model_type)
    if args.dataset in ['java250','python800']:
        train_samples,valid_samples,test_samples,token_vocabsize,type_vocabsize=get_spt_dataset(data=args.dataset,mislabeled_rate=args.noise_rate)
    else:
        raise NotImplementedError
    trainset=GraphClassificationDataset(train_samples)
    validset=GraphClassificationDataset(valid_samples)
    testset=GraphClassificationDataset(test_samples)
    print(len(trainset),len(validset),len(testset))

    model=GNN_codenet(256,num_classes,num_layers=5,token_vocabsize=token_vocabsize,type_vocabsize=type_vocabsize,model=args.model_type).to(device)
    train_dataloader=GraphDataLoader(trainset,batch_size=args.batch_size,shuffle=True)
    valid_dataloader=GraphDataLoader(validset,batch_size=args.batch_size,shuffle=False)
    test_dataloader=GraphDataLoader(testset,batch_size=args.batch_size,shuffle=False)
else:
    if args.dataset in ['java250','python800']:
        train_samples,valid_samples,test_samples=read_codenetdata(dataname=args.dataset,mislabeled_rate=args.noise_rate,noise_pattern=args.noise_pattern)

    trainset=ClassificationDataset(tokenizer,args,train_samples)
    validset=ClassificationDataset(tokenizer,args,valid_samples)
    testset=ClassificationDataset(tokenizer,args,test_samples)
    print(len(trainset),len(validset),len(testset))

    #choose classifier: pre-trained or lstm
    #model=bert_classifier_self(model_encoder,encoder_config,tokenizer,args)
    model=bert_and_linear_classifier(model_encoder.roberta,encoder_config,tokenizer,args,num_classes)
    if args.model_type=='lstm':
        model=lstm_classifier(encoder_config.vocab_size,128,128,num_classes)
    #model=lstm_classifier(encoder_config.vocab_size,128,128,num_classes)
    model=model.to(device)

    train_dataloader = DataLoader(trainset, shuffle=True, batch_size=args.batch_size,num_workers=0)
    valid_dataloader = DataLoader(validset, shuffle=False, batch_size=args.batch_size,num_workers=0)
    test_dataloader = DataLoader(testset, shuffle=False, batch_size=args.batch_size,num_workers=0)

optimizer = optim.AdamW(model.parameters(), lr=args.lr)
criterion=nn.CrossEntropyLoss()
args.max_steps=args.epochs*len(train_dataloader) #num_epochs*num_batches
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
print('fp16:',args.fp16)
if args.fp16:
    scaler = GradScaler()
    
    
def evaluate(model,dataloader):
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[] 
    labels=[]

    bar=tqdm(dataloader)
    for batch in bar:
        inputs = batch[0].to(device)        
        label=batch[1].to(device) 
        with torch.no_grad():
            #lm_loss,logit = model(inputs,label)
            #eval_loss += lm_loss.mean().item()
            logit=model(inputs)
            eval_loss=F.cross_entropy(logit, label.long(), reduction='mean')
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1

        bar.set_description("loss {}".format(eval_loss.item()))

    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    #preds=logits[:,0]>0.5 #binary
    preds=np.argmax(logits,1)
    eval_acc=np.mean(labels==preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
            
    result = {
        "eval_loss": float(perplexity),
        "eval_acc":round(eval_acc,4),
    }
    return result
  
  
global_step=0
tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
best_mrr=0.0
best_acc=0.0

best_valid_acc=0

model.zero_grad()
for epoch in range(args.epochs):
    bar = tqdm(train_dataloader,total=len(train_dataloader))
    tr_num=0
    train_loss=0
    training_original_labels=[]
    training_noisy_labels=[]
    training_meta_weights=[]
    model.train()
    for step, batch in enumerate(bar):
        #print(batch)
        inputs = batch[0].to(device)        
        labels=batch[1].to(device) 
        original_labels=batch[2].to(device)
        #print(inputs.size(),labels.size(),original_labels.size())
        
        outputs=model(inputs)
        loss=F.cross_entropy(outputs, labels.long(), reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        tr_loss += loss.item()
        tr_num+=1
        train_loss+=loss.item()
        if avg_loss==0:
            avg_loss=tr_loss
        avg_loss=round(train_loss/tr_num,5)
        bar.set_description("epoch {} loss {}".format(epoch+1,loss.item()))
        
        optimizer.step()
        scheduler.step()
        global_step += 1
        output_flag=True
        avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

        tr_nb=global_step
        
    print('----validation----')
    valid_res=evaluate(model,valid_dataloader)
    print(valid_res)
    valid_acc=valid_res['eval_acc']
    if valid_acc>best_valid_acc:
        best_valid_acc=valid_acc
        print('best epoch')
        print('----test----')
        test_res=evaluate(model,test_dataloader)
        print(test_res)
