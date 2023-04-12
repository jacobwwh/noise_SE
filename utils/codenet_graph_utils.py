import csv
import random
import torch
import dgl
import dgl.function as fn
from dgl.data import DGLDataset
from torch.utils.data import Dataset

pythonpath='/mnt/nvme/wenhan/codenet/Project_CodeNet_Python800_spts/'
c1000path='/mnt/nvme/wenhan/codenet/Project_CodeNet_C++1000_spts/'
c1400path='/mnt/nvme/wenhan/codenet/Project_CodeNet_C++1400_spts/'
javapath='/mnt/nvme/wenhan/codenet/Project_CodeNet_Java250_spts/'

def random_noise_label(num_classes,correct_label):
    label=random.randint(0,num_classes-1)
    while label==correct_label:
        label=random.randint(0,num_classes-1)
    return label


def get_spt_dataset(bidirection=True, virtual=False,edgetype=False,next_token=True,data='java250',mislabeled_rate=0.2,noise_pattern='random',noisy_test=False):
    assert data in ['java250','c++1000','c++1400','python800']
    print('next_token:',next_token)
    print('noise rate:',mislabeled_rate,'noise pattern:',noise_pattern)
    if data=='java250':
        datapath=javapath
        num_classes=250
    elif data=='c++1000':
        datapath=c1000path
        num_classes=1000
    elif data=='c++1400':
        datapath=c1400path
        num_classes=1400
    elif data=='python800':
        datapath=pythonpath
        num_classes=800
    
    edgepath=datapath+'edge.csv'
    labelpath=datapath+'graph-label.csv'
    nodepath=datapath+'node-feat.csv'
    edgenumpath=datapath+'num-edge-list.csv'
    nodenumpath=datapath+'num-node-list.csv'

    numnodes=[]
    numedges=[]
    nodefeats=[]
    is_tokens=[]
    token_ids=[]
    rule_ids=[]
    edges=[]
    labels=[]
    token_vocabsize=0
    type_vocabsize=0
    with open(nodenumpath) as f:
        f_csv=csv.reader(f)
        for row in f_csv:
            numnodes.append(int(row[0]))
    with open(edgenumpath) as f:
        f_csv=csv.reader(f)
        for row in f_csv:
            numedges.append(int(row[0]))
    with open(labelpath) as f:
        f_csv=csv.reader(f)
        for row in f_csv:
            labels.append(int(row[0]))
    print(len(numnodes),len(numedges),len(labels))
    with open(edgepath) as f:
        f_csv=csv.reader(f)
        for row in f_csv:
            source,target=row
            source,target=int(source),int(target)
            edges.append([source,target])
    print(len(edges))
    with open(nodepath) as f:
        f_csv=csv.reader(f)
        for row in f_csv:
            is_token,token_type,rule_type,is_reserved=row
            token_type,rule_type=int(token_type),int(rule_type)
            if token_type>token_vocabsize:
                token_vocabsize=token_type
            if rule_type>type_vocabsize:
                type_vocabsize=rule_type
            nodefeats.append([token_type,rule_type])
            token_ids.append(token_type)
            rule_ids.append(rule_type)
            is_tokens.append(int(is_token))
    print(len(nodefeats))
    
    all_graphdata=[]
    train_data=[]
    dev_data=[]
    test_data=[]
    for i in range(num_classes):
        all_graphdata.append([])
        train_data.append([])
        dev_data.append([])
        test_data.append([])
    graph_nodestart=0
    graph_edgestart=0
    if edgetype==True:
        for i in range(len(labels)):
            num_node,num_edge,graph_label=numnodes[i],numedges[i],labels[i]
            graph_edge=edges[graph_edgestart:graph_edgestart+num_edge]
            
            graph_istoken=is_tokens[graph_nodestart:graph_nodestart+num_node]
            token_ids=[i for i in range(num_node) if graph_istoken[i]==1]
            
            targets,sources=list(zip(*graph_edge)) #from child to parent
            targets,sources=list(targets),list(sources)
            edge_types=[0]*len(targets)+[1]*len(targets) #0: child->parent, 1: parent->child, 2: nextsib, 3: prevsib
            targets,sources=targets+sources,sources+targets #add parent->child edges
            for i in range(len(graph_edge)): #add sibling edges
                parentid,childid=graph_edge[i]
                if i>0 and parentid==graph_edge[i-1][0]: #have same parent
                    sources.append(graph_edge[i-1][1])
                    targets.append(graph_edge[i][1])
                    edge_types.append(2)
                    sources.append(graph_edge[i][1])
                    targets.append(graph_edge[i-1][1])
                    edge_types.append(3)
            g=dgl.graph((sources,targets))
            graph_tokens=torch.tensor(token_ids[graph_nodestart:graph_nodestart+num_node])
            graph_rules=torch.tensor(rule_ids[graph_nodestart:graph_nodestart+num_node])
            edge_types=torch.tensor(edge_types)
            g.ndata['token']=graph_tokens
            g.ndata['type']=graph_rules
            g.edata['etype']=edge_types
            graph_nodestart+=num_node
            graph_edgestart+=num_edge
            all_graphdata[graph_label].append({'code':g,'label':graph_label,'original_label':graph_label})
    else:
        for i in range(len(labels)):
            num_node,num_edge,graph_label=numnodes[i],numedges[i],labels[i]
            graph_edge=edges[graph_edgestart:graph_edgestart+num_edge]
            targets,sources=list(zip(*graph_edge)) #from child to parent
            if next_token:
                leaf_ids=[i for i in range(num_node) if i not in targets]
            if bidirection==True: # bidirectional graph for gnn
                targets,sources=targets+sources,sources+targets
                if next_token:
                    nexttoken_srcs=leaf_ids[:-1]
                    nexttoken_tgts=leaf_ids[1:]
                    nexttoken_srcs,nexttoken_tgts=nexttoken_srcs+nexttoken_tgts,nexttoken_tgts+nexttoken_srcs
                    nexttoken_srcs,nexttoken_tgts=tuple(nexttoken_srcs),tuple(nexttoken_tgts)
                    targets,sources=targets+nexttoken_tgts,sources+nexttoken_srcs

            targets,sources=torch.tensor(targets),torch.tensor(sources)
            g=dgl.graph((sources,targets))
            graph_tokens=torch.tensor(token_ids[graph_nodestart:graph_nodestart+num_node])
            graph_rules=torch.tensor(rule_ids[graph_nodestart:graph_nodestart+num_node])
            g.ndata['token']=graph_tokens
            g.ndata['type']=graph_rules
            graph_nodestart+=num_node
            graph_edgestart+=num_edge
            all_graphdata[graph_label].append({'code':g,'label':graph_label,'original_label':graph_label})

    #simple data split
    print(len(all_graphdata))
    for i in range(num_classes):
        for j in range(len(all_graphdata[i])):
            if j%5==3:
                dev_data[i].append(all_graphdata[i][j])
            elif j%5==4:
                test_data[i].append(all_graphdata[i][j])
            else:
                train_data[i].append(all_graphdata[i][j])
    print(len(train_data[0]),len(dev_data[0]),len(test_data[0]))

    token_vocabsize+=2
    type_vocabsize+=2
    print(token_vocabsize,type_vocabsize)
    
    #add noise
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
        if noisy_test:
            print('create noisy valid/test set')
            for i in range(num_classes):
                mislabeld_dev_idx=random.sample(range(len(dev_data[i])),int(len(dev_data[i])*mislabeled_rate))
                mislabeld_dev_idx=set(mislabeld_dev_idx)
                for j in range(len(dev_data[i])):
                    if j in mislabeld_dev_idx:
                        if noise_pattern=='random':
                            dev_data[i][j]['label']=random_noise_label(num_classes,i)
                        elif noise_pattern=='flip':
                            dev_data[i][j]['label']=(i+1)%num_classes
                        elif noise_pattern=='pair':
                            if i%2==0:
                                dev_data[i][j]['label']=i+1
                            else:
                                dev_data[i][j]['label']=i-1
                        else:
                            print('undefined noise pattern!')
                            quit()
                mislabeld_test_idx=random.sample(range(len(test_data[i])),int(len(test_data[i])*mislabeled_rate))
                mislabeld_test_idx=set(mislabeld_test_idx)
                for j in range(len(test_data[i])):
                    if j in mislabeld_test_idx:
                        if noise_pattern=='random':
                            test_data[i][j]['label']=random_noise_label(num_classes,i)
                        elif noise_pattern=='flip':
                            test_data[i][j]['label']=(i+1)%num_classes
                        elif noise_pattern=='pair':
                            if i%2==0:
                                test_data[i][j]['label']=i+1
                            else:
                                test_data[i][j]['label']=i-1
                        else:
                            print('undefined noise pattern!')
                            quit()
    return train_data,dev_data,test_data,token_vocabsize,type_vocabsize



class GraphClassificationDataset(Dataset):
    def __init__(self, classified_data):
        self.examples = []
        for js_list in classified_data:
            for js in js_list:
                self.examples.append(js)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.examples[i]['original_label'] is not None:
            return self.examples[i]['code'],torch.tensor(self.examples[i]['label']),torch.tensor(self.examples[i]['original_label'])
        else:
            return self.examples[i]['code'],torch.tensor(self.examples[i]['label'])
