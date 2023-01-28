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
                          
from utils.poj_utils import ClassificationDataset,generate_pojdata
from utils.codenet_utils import read_codenetdata
from model.bert import bert_classifier_self,lstm_classifier

from utils.codenet_graph_utils import get_spt_dataset,GraphClassificationDataset
from dgl.dataloading import GraphDataLoader
from model.gnn import GNN_codenet

import cleanlab
from cleanlab.filter import find_label_issues,find_predicted_neq_given
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='java250')

parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.5)

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

assert args.dataset in ['poj','java250','python800']
assert args.model_type in ['codebert','graphcodebert','codet5','unixcoder','gcn','gin','ggnn','hgt']
if args.dataset=='poj':
    num_classes=104 #poj
elif args.dataset=='java250':
    num_classes=250 #codenet java250
elif args.dataset=='python800':
    num_classes=800 #codenet python800

if args.model_type not in ['gcn','gin','ggnn','hgt']:
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
    train_dataloader=GraphDataLoader(trainset,batch_size=args.batch_size,shuffle=False)
    valid_dataloader=GraphDataLoader(validset,batch_size=args.batch_size,shuffle=False)
    test_dataloader=GraphDataLoader(testset,batch_size=args.batch_size,shuffle=False)
else:
    if args.dataset=='poj':
        train_samples,valid_samples,test_samples=generate_pojdata(mislabeled_rate=args.noise_rate)
    if args.dataset in ['java250','python800']:
        train_samples,valid_samples,test_samples=read_codenetdata(dataname=args.dataset,mislabeled_rate=args.noise_rate)
        
    trainset=ClassificationDataset(tokenizer,args,train_samples)
    validset=ClassificationDataset(tokenizer,args,valid_samples)
    testset=ClassificationDataset(tokenizer,args,test_samples)
    print(len(trainset),len(validset),len(testset))

    #choose classifier: pre-trained or lstm
    model=bert_classifier_self(model_encoder,encoder_config,tokenizer,args)
    #model=lstm_classifier(encoder_config.vocab_size,128,128,num_classes)
    model=model.to(device)

    train_dataloader = DataLoader(trainset, shuffle=False, batch_size=args.batch_size,num_workers=0)
    valid_dataloader = DataLoader(validset, shuffle=False, batch_size=args.batch_size,num_workers=0)
    test_dataloader = DataLoader(testset, shuffle=False, batch_size=args.batch_size,num_workers=0)

#model for re-training on selected clean samples
retrain_model=copy.deepcopy(model)

optimizer = optim.AdamW(retrain_model.parameters(), lr=args.lr)
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




#load models trained on noisy data
if args.model_type in ['codebert','graphcodebert','codet5','unixcoder']:
    save_path='saved_models/'+args.model_type+'_'+args.dataset+'_'+str(args.noise_rate)+'.bin'
    print('load model from:',save_path)
    model.load_state_dict(torch.load(save_path))


def find_noisy(model,dataloader,trainset):
    model.eval()
    logits=[] 
    labels=[]
    original_labels=[]

    bar=tqdm(dataloader)
    for i,batch in enumerate(bar):
        inputs = batch[0].to(device)        
        label=batch[1].to(device) 
        original_label=batch[2].to(device)
        with torch.no_grad():
            #lm_loss,logit = model(inputs,label)
            #eval_loss += lm_loss.mean().item()
            logit=model(inputs)
            logit=F.softmax(logit,dim=1) #convert logit to probs
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        #if i>1:
            #break
    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)

    ordered_label_issues = find_label_issues(
        labels=labels,
        pred_probs=logits, # out-of-sample predicted probabilities from any model
        return_indices_ranked_by=None,  #or 'self_confidence'
        filter_by='prune_by_noise_rate'
        )
    #print(len(ordered_label_issues))
    #print("The Index of Error Samples are: {}".format(",".join([str(ele) for ele in ordered_label_issues])))

    #print(labels[:10])
    #print(np.argmax(logits,1)[:10])
    # a simple baseline: directly use logits to predict noises
    label_issues_base=find_predicted_neq_given(labels,logits) 
    #label_issues_base=np.argmax(logits, axis=1) != np.asarray(labels)

    ground_truth_noises=[]
    for sample in trainset:
        ground_truth_noises.append(sample[1]!=sample[2])
    ground_truth_noises=np.array(ground_truth_noises)

    acc=np.mean(ground_truth_noises==ordered_label_issues)
    precision=precision_score(ground_truth_noises,ordered_label_issues)
    recall=recall_score(ground_truth_noises,ordered_label_issues)
    f1=f1_score(ground_truth_noises,ordered_label_issues)
    confident_res={'acc':acc,'precision':precision,'recall':recall,'f1':f1}
    print('confident learning:',confident_res)

    acc_base=np.mean(ground_truth_noises==label_issues_base)
    precision_base=precision_score(ground_truth_noises,label_issues_base)
    recall_base=recall_score(ground_truth_noises,label_issues_base)
    f1_base=f1_score(ground_truth_noises,label_issues_base)
    base_res={'acc':acc_base,'precision':precision_base,'recall':recall_base,'f1':f1_base}
    print('baseline:',base_res)
    return ordered_label_issues,label_issues_base


def filter_by_predicted(trainset,label_issues):
    """filter the training set by predicted noisy labels"""
    print(len(trainset))
    print(label_issues.shape)
    new_examples=[]
    for i in range(len(trainset)):
        #print(trainset[i],label_issues[i])
        if(label_issues[i]==False):
            new_examples.append(trainset[i])
    #trainset.examples=new_examples #unnecessary
    #print(len(trainset))
    return new_examples


label_issues,label_issues_base=find_noisy(model,train_dataloader,trainset)
filtered_trainset=filter_by_predicted(trainset,label_issues) #label_issues or label_issues_base
filtered_train_loader=DataLoader(filtered_trainset, shuffle=True, batch_size=args.batch_size,num_workers=0)




#retrain the model on filtered training set
global_step=0
tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
best_mrr=0.0
best_acc=0.0

best_valid_acc=0

for epoch in range(args.epochs):
    bar = tqdm(filtered_train_loader,total=len(filtered_train_loader))
    tr_num=0
    train_loss=0
    training_original_labels=[]
    training_noisy_labels=[]
    retrain_model.train()
    for step, batch in enumerate(bar):
        #print(batch)
        inputs = batch[0].to(device)        
        labels=batch[1].to(device) 
        original_labels=batch[2].to(device)
        #print(inputs.size(),labels.size(),original_labels.size())
        
        outputs=retrain_model(inputs)
        loss=F.cross_entropy(outputs, labels.long(), reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(retrain_model.parameters(), 0.5)
        
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
    valid_res=evaluate(retrain_model,valid_dataloader)
    print(valid_res)
    valid_acc=valid_res['eval_acc']
    if valid_acc>best_valid_acc:
        best_valid_acc=valid_acc
        print('best epoch')
        print('----test----')
        test_res=evaluate(retrain_model,test_dataloader)
        print(test_res)

