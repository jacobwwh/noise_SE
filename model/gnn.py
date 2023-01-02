import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GlobalAttentionPooling, GraphConv, GINConv, GatedGraphConv, HGTConv


class GNN_codenet(torch.nn.Module):
    def __init__(self, dim_model, n_classes, num_layers, token_vocabsize, type_vocabsize, dropout=0.2,model='gcn'):
        super(GNN_codenet, self).__init__()
        self.dim_model=dim_model
        self.dropout = torch.nn.Dropout(dropout)
        self.layers=nn.ModuleList()
        self.model=model
        if model=='ggnn':
            self.layers.append(GatedGraphConv(dim_model,dim_model,n_steps=4,n_etypes=4))
        else:
            for i in range(num_layers):
                if model=='gcn':
                    self.layers.append(GraphConv(dim_model,dim_model,activation=F.relu))
                elif model=='gin':
                    gin_mlp = nn.Sequential(nn.Linear(dim_model, 2*dim_model), nn.BatchNorm1d(2*dim_model), nn.ReLU(), nn.Linear(2*dim_model, dim_model))
                    #gin_mlp = nn.Sequential(nn.Linear(dim_model, 2*dim_model), nn.ReLU(), nn.Linear(2*dim_model, dim_model))
                    self.layers.append(GINConv(gin_mlp, 'sum',learn_eps=True))
                elif model=='hgt':
                    self.num_heads=4
                    #self.layers.append(HGTConv(dim_model, dim_model//self.num_heads,self.num_heads,num_ntypes=2,num_etypes=10,use_norm=True))
        self.token_embeddings=nn.Embedding(token_vocabsize,dim_model//2)
        self.type_embeddings=nn.Embedding(type_vocabsize,dim_model//2)
        print(self.token_embeddings)
        print(self.type_embeddings)
        self.classifier=nn.Linear(dim_model,n_classes)
        self.pooling=GlobalAttentionPooling(nn.Linear(dim_model,1))
        self.pool_mode='attention'
        print(self.pool_mode)

    def forward(self, batch,root_ids=None,etypes=None):
        batch.ndata['h']=torch.cat([self.type_embeddings(batch.ndata['type']),self.token_embeddings(batch.ndata['token'])],dim=1)
        h=batch.ndata['h']
        if self.model=='ggnn':
            if self.model=='ggnn':
                etypes=torch.zeros([batch.num_edges()],device=batch.device)
            elif self.model=='ggnn-typed':
                etypes=batch.edata['etype']
            h=self.layers[0](batch,h,etypes=etypes)
        else:
            if self.model=='hgt':
                ntypes=torch.zeros(batch.num_nodes(),dtype=torch.long).to(batch.device)
                etypes=torch.zeros(batch.num_edges(),dtype=torch.long).to(batch.device)
                for i, layer in enumerate(self.layers):
                    #if i!=0:
                        #h=self.dropout(h)    
                    h=layer(batch,h,ntype=ntypes,etype=etypes,presorted=True)
            else:
                for i, layer in enumerate(self.layers):
                    if i!=0:
                        h=self.dropout(h)    
                    h=layer(batch,h)
        batch.ndata['h']=h
        if self.pool_mode=='root':
            assert root_ids is not None
            batch_pred=batch.ndata['h'][root_ids]
        else:
            batch_pred=self.pooling(batch,batch.ndata['h'])

        batch_pred=self.classifier(batch_pred)
        return batch_pred
