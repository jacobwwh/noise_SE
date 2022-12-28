import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class bert_classifier_self(nn.Module):   
    '''a binary classifier for bert-like models'''
    def __init__(self, encoder,config,tokenizer,args):
        super(bert_classifier_self, self).__init__()
        self.encoder = encoder #transformer.BertForSequenceClassification
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        
    def forward(self, input_ids=None,labels=None,return_h=False): 
        raw_output=self.encoder(input_ids,attention_mask=input_ids.ne(1),output_hidden_states=True)
        outputs=raw_output[0] #(batch_size, config.num_labels)

        logits=outputs
        hidden_states=raw_output.hidden_states
        pooled_outputs=hidden_states[-1][:,0,:]
        if return_h:
            return logits,pooled_outputs

        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,logits
        else:
            return logits
          
          
class lstm_classifier(nn.Module):   
    def __init__(self,vocab_size,embedding_dim,hidden_dim,n_classes,n_layers=1,bidirectional=True,dropout=0.0):       
        super(lstm_classifier,self).__init__()
        
        # Embedding layer converts integer sequences to vector sequences
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        
        # LSTM layer process the vector sequences 
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            dropout = dropout,
                            batch_first = True
                           )
        
        # Dense layer to predict 
        self.transform = nn.Linear(hidden_dim * 2,hidden_dim,bias=False)
        self.fc = nn.Linear(hidden_dim,n_classes)
        # Prediction activation function
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self,text,text_lengths=None,return_h=False):
        embedded = self.embedding(text)
        text_lengths=text.ne(1).sum(1)
        
        # Thanks to packing, LSTM don't see padding tokens 
        # and this makes our model better
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(),batch_first=True,enforce_sorted=False)
        
        with torch.backends.cudnn.flags(enabled=False):
            packed_output,(hidden_state,cell_state) = self.lstm(packed_embedded)
        
        # Concatenating the final forward and backward hidden states
        hidden = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)
        hidden=self.transform(hidden)
        dense_outputs=self.fc(hidden)

        #Final activation function
        outputs=self.sigmoid(dense_outputs)
        if return_h:
            return dense_outputs,hidden     
        return dense_outputs
