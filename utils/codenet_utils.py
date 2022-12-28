import os
import json
import random
import torch
import random
from tqdm import tqdm


pythonpath='/mnt/nvme/wenhan/codenet/Project_CodeNet_Python800/'
c1000path='/mnt/nvme/wenhan/codenet/Project_CodeNet_C++1000/'
c1400path='/mnt/nvme/wenhan/codenet/Project_CodeNet_C++1400/'
javapath='/mnt/nvme/wenhan/codenet/Project_CodeNet_Java250/'

datapaths={'java250':javapath,'c++1000':c1000path,'c++1400':c1400path,'python800':pythonpath}


def random_noise_label(num_classes,correct_label):
    label=random.randint(0,num_classes-1)
    while label==correct_label:
        label=random.randint(0,num_classes-1)
    return label


def read_codenetdata(dataname,mislabeled_rate=0.2,noise_pattern='random'):
    datapath=datapaths[dataname]
    if dataname=='java250':
        num_classes=250
    elif dataname=='python800':
        num_classes=800
    print(datapath)
    classes_paths=[]
    all_data=[]
    train_data=[]
    dev_data=[]
    test_data=[]
    for i in range(num_classes):
        all_data.append([])
        train_data.append([])
        dev_data.append([])
        test_data.append([])
    
    for filename in os.listdir(datapath):
        classes_paths.append(datapath+filename)
    for i,path in enumerate(classes_paths):
        for filename in os.listdir(path):
            with open(path+'/'+filename) as f:
                sourcecode=f.read()
                data={'code':sourcecode,'label':i,'original_label':i}
                all_data[i].append(data)

    for i in range(num_classes):
        for j in range(len(all_data[i])):
            if j%5==3:
                dev_data[i].append(all_data[i][j])
            elif j%5==4:
                test_data[i].append(all_data[i][j])
            else:
                train_data[i].append(all_data[i][j])
    
    if mislabeled_rate>0:
        for i in range(num_classes):
            mislabeld_train_idx=random.sample(range(len(train_data[i])),int(len(train_data[i])*mislabeled_rate))
            mislabeld_train_idx=set(mislabeld_train_idx)
            for j in range(len(train_data[i])):
                if j in mislabeld_train_idx:
                    if noise_pattern=='random':
                        train_data[i][j]['label']=random_noise_label(num_classes,i)
                    elif noise_pattern=='flip':
                        train_data[i][j]['label']=(i+1)%num_classes
    return train_data,dev_data,test_data
