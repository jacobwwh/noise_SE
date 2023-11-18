import argparse
import time
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

from robusttrainer.utils import DAverageMeter
from robusttrainer.class_prototypes import get_prototypes
from sklearn.mixture import GaussianMixture

import logging

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='java250')

#robusttrainer args
parser.add_argument("--seed", default=42, type=int, help="Random seed (for RobustTrainer).")
parser.add_argument("--warmup_epochs", default=5, type=int, help="Warm-up epochs before apply RobustTrainer.")
parser.add_argument("--nb_prototypes", default=1, type=int, help="Number of prototypes per class?")
parser.add_argument('--temperature', type = float, help = '', default = 1)
parser.add_argument('--w_ce', type = float, help = 'Weight of cross entropy loss', default = 1)
parser.add_argument('--w_cl', type = float, help = 'Weight of contrastive loss', default = 1)

parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.5)
parser.add_argument("--noise_pattern", default="random", type=str, help="Noise pattern(random/flip/pair).")

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
parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--epochs", default=100, type=int, help="Training epochs.")
parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
args=parser.parse_args()

logging.basicConfig(filename='logs/robusttrainer_'+args.dataset+'_'+str(args.noise_rate)+args.noise_pattern+'1.log',
                        level = logging.INFO)
logger = logging.getLogger()
logger.info(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert args.dataset in ['java250','python800']
assert args.model_type in ['codebert','graphcodebert','unixcoder','gin','lstm']
if args.dataset=='java250':
    num_classes=250 #codenet java250
elif args.dataset=='python800':
    num_classes=800 #codenet python800

args.nb_classes=num_classes #for robusttrainer

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
        train_samples,valid_samples,test_samples,token_vocabsize,type_vocabsize=get_spt_dataset(data=args.dataset,mislabeled_rate=args.noise_rate,noise_pattern=args.noise_pattern)
    else:
        raise NotImplementedError
    trainset=GraphClassificationDataset(train_samples)
    validset=GraphClassificationDataset(valid_samples)
    testset=GraphClassificationDataset(test_samples)
    print(len(trainset),len(validset),len(testset))

    model=GNN_codenet(256,num_classes,num_layers=5,token_vocabsize=token_vocabsize,type_vocabsize=type_vocabsize,model=args.model_type).to(device)
    train_dataloader=GraphDataLoader(trainset,batch_size=args.batch_size,shuffle=True)
    train_dataloader_eval=GraphDataLoader(trainset,batch_size=args.batch_size,shuffle=False)
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
    model=model.to(device)

    train_dataloader = DataLoader(trainset, shuffle=True, batch_size=args.batch_size,num_workers=0)
    train_dataloader_eval=DataLoader(trainset, shuffle=False, batch_size=args.batch_size,num_workers=0) #for calculating features for robusttrainer
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


class FeatureLearning(object):
    def __init__(self, args):
        self.args = args
        self.model = model

        self.CE = nn.CrossEntropyLoss().cuda()
        self.NLL = nn.NLLLoss().cuda()
        #self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        self.log_file = None
        self.logger = None

        self.prototypes = None
        self.dataset_cache = {
            "clean_idx": None,
        }

RobustTrainer_logger=FeatureLearning(args) #keep prototypes and clean_idx




def select_clean_samples(x, y, args):
    """
    Clean the dataset
    :param x: features of all examples in the dataset
    :param y: original labels
    :param args: hyper-parameter
    :return:
    """
    prototypes=get_prototypes(x,y,args)  #list of size(num_classes, args.nb_prototypes)
    prototypes=np.vstack(prototypes)
    print(prototypes.shape)

    similarities_proto = x.dot(prototypes.T)
    similarities_class = np.zeros((x.shape[0], args.nb_classes), dtype=np.float64)
    print(similarities_class)

    for i in range(args.nb_classes):
        similarities_class[:, i] = np.mean(
            similarities_proto[:, i * args.nb_prototypes:(i + 1) * args.nb_prototypes], axis=1)
    # select the samples by GMM
    clean_set = []
    for i in range(args.nb_classes):
        class_idx = np.where(y == i)[0]
        class_sim = similarities_proto[class_idx, i]
        # split the dataset using GMM
        class_sim = class_sim.reshape((-1, 1))
        # gm = GaussianMixture(n_components=2, random_state=args.seed).fit(class_sim)
        gm = GaussianMixture(n_components=2, random_state=args.seed).fit(class_sim)
        class_clean_idx = np.where(gm.predict(class_sim) == gm.means_.argmax())[0]
        clean_set.extend(class_idx[class_clean_idx])

    print('number of selected clean samples:',len(clean_set))
    return x[clean_set], y[clean_set], clean_set


def run_train_epoch(model, data_loader, current_epoch):
    # train the model with both contrastive and CE loss
    model.train()
    bar = tqdm(data_loader,total=len(data_loader))
    for step, batch in enumerate(bar):
        inputs = batch[0].to(device)        
        labels=batch[1].to(device) 
        original_labels=batch[2].to(device)
        record = {}

        #train with contrastive loss
        logit,feature=model(inputs,return_h=True)  
        class_prototypes=copy.deepcopy(RobustTrainer_logger.prototypes)
        class_prototypes = torch.from_numpy(class_prototypes).cuda()
        logits_proto = torch.mm(feature, class_prototypes.t()) / args.temperature
        softmax_proto = F.softmax(logits_proto, dim=1)
        prob_proto = torch.zeros((softmax_proto.shape[0], args.nb_classes), dtype=torch.float64).cuda()
        for i in range(args.nb_classes):
            prob_proto[:, i] = torch.sum(
                softmax_proto[:, i * args.nb_prototypes: (i + 1) * args.nb_prototypes], dim=1)
        # contrastive loss
        cl_loss = RobustTrainer_logger.NLL(torch.log(prob_proto + 1e-5), labels)
        record['loss_contrastive'] = cl_loss.item()
        # classification loss
        ce_loss = RobustTrainer_logger.CE(logit, labels)
        record['loss'] = ce_loss.item()

        loss = args.w_ce * ce_loss + args.w_cl * cl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        bar.set_description("epoch {} loss {}".format(current_epoch+1,record))





global_step=0
tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
best_valid_acc=0
best_test_res={}
best_epoch=0

for epoch in range(args.epochs):
    if epoch<args.warmup_epochs: #naive training for warmup
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
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
            
        print('----warm-up validation----')
        valid_res=evaluate(model,valid_dataloader)
        print(valid_res)
        valid_acc=valid_res['eval_acc']
        if valid_acc>best_valid_acc:
            best_valid_acc=valid_acc
            print('best epoch')
            print('----warm-up test----')
            test_res=evaluate(model,test_dataloader)
            print(test_res)

    else: #robusttrainer training
        #extract features for all training samples
        t_start=time.time()
        model.eval()
        labels=[]
        features=[]
        bar=tqdm(train_dataloader_eval)
        for batch in bar:
            inputs = batch[0].to(device)        
            label=batch[1].to(device) #(noisy)
            with torch.no_grad():
                logit,feature=model(inputs,return_h=True)  
            features.append(feature.cpu().numpy())
            labels.append(label.cpu().numpy())
        features=np.concatenate(features,0)
        labels=np.concatenate(labels,0)
        print(features.shape,labels.shape)

        #select clean samples
        clean_feat, clean_y, clean_idx=select_clean_samples(features,labels,args) #clean_idx is not sorted by index yet
        clean_examples=[trainset[i] for i in clean_idx]
        for j in range(10):
            print(clean_idx[j],clean_examples[j][1],clean_examples[j][2],clean_examples[j][3])
        train_dataloader_selected=DataLoader(clean_examples,batch_size=args.batch_size,shuffle=True)

        # refine the class prototypes using the newly selected clean samples
        class_prototypes = get_prototypes(clean_feat, clean_y, args)
        class_prototypes = np.vstack(class_prototypes)
        RobustTrainer_logger.prototypes = class_prototypes

        #train on selected clean samples
        run_train_epoch(model,train_dataloader_selected,epoch)
        t_end=time.time()
        logger.info("time for epoch: {}".format(t_end-t_start))

        print('----RobustTrainer validation----')
        valid_res=evaluate(model,valid_dataloader)
        print(valid_res)
        logger.info("epoch {}".format(epoch+1))
        logger.info("validation results {}".format(valid_res))
        valid_acc=valid_res['eval_acc']
        if epoch>best_epoch+15:
            print('early stop')
            break
        if valid_acc>best_valid_acc:
            best_valid_acc=valid_acc
            print('best epoch')
            print('----RobustTrainer test----')
            test_res=evaluate(model,test_dataloader)
            print(test_res)
            best_test_res=test_res
            best_epoch=epoch
            
            logger.info("epoch {}".format(epoch+1))
            logger.info("test results {}".format(test_res))

print('best epoch',best_epoch+1,'test results:',best_test_res)
        
