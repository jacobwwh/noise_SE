import os
import json
import random
import torch
from torch.utils.data import DataLoader, Dataset
from utils.dataset_utils import convert_examples_to_features_noisyclassification

data_dir='poj104/'
num_classes=104

def random_noise_label(num_classes,correct_label):
    label=random.randint(0,num_classes-1)
    while label==correct_label:
        label=random.randint(0,num_classes-1)
    return label
    
    
def generate_pojdata(mislabeled_rate=0.0,noise_pattern='random'):
    train_data=[]
    valid_data=[]
    test_data=[]
    for i in range(num_classes):
        train_data.append([])
        valid_data.append([])
        test_data.append([])
    for i in range(num_classes):
        for idx,path in enumerate(os.listdir(os.path.join(data_dir,str(i+1)))):
            with open(os.path.join(data_dir,str(i+1),path)) as f:
                code=f.read()
                if idx%10==7:
                    valid_data[i].append({'code':code,'label':i,'original_label':i})
                elif idx%10==8 or idx%10==9:
                    test_data[i].append({'code':code,'label':i,'original_label':i})
                else:
                    train_data[i].append({'code':code,'label':i,'original_label':i})

    if(mislabeled_rate>0):
        for i in range(num_classes):
            mislabeld_train_idx=random.sample(range(len(train_data[i])),int(len(train_data[i])*mislabeled_rate))
            mislabeld_train_idx=set(mislabeld_train_idx)
            for j in range(len(train_data[i])):
                if j in mislabeld_train_idx:
                    if noise_pattern=='random':
                        train_data[i][j]['label']=random_noise_label(num_classes,i)
                    elif noise_pattern=='flip':
                        train_data[i][j]['label']=(i+1)%num_classes

    return train_data,valid_data,test_data
    
    
class ClassificationDataset(Dataset):
    def __init__(self, tokenizer, args, classified_data):
        self.examples = []
        for js_list in classified_data:
            for js in js_list:
                self.examples.append(convert_examples_to_features_noisyclassification(js,tokenizer,args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.examples[i].original_label is not None:
            return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label),torch.tensor(self.examples[i].original_label)
        else:
            return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)
