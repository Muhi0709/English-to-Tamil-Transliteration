import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence


class encoder(nn.Module):
    def __init__(self,hidden_size,num_of_hidden_layers,dict_size,embedding_size,cell_type="rnn",bidirectional=False,dropout=0):
        super(encoder,self).__init__()
        self.hidden_size=hidden_size
        self.num_of_hidden_layers=num_of_hidden_layers
        self.dict_size=dict_size
        self.embedding_size=embedding_size
        self.cell_type=cell_type
        self.bidirectional= bidirectional
        self.dropout=dropout


        self.dropout_layer=None
        self.h0=None
        self.c0=None
        self.embedded_x=None
        self.hidden_output = None
        self.hidden_last = None
        self.cell_last = None
        self.cell=None
        self.output=None

        self.embedding = nn.Embedding(dict_size,embedding_size)
        self.dropout_layer = nn.Dropout(self.dropout)

        if self.cell_type == "rnn":
            self.cell = nn.RNN(embedding_size,hidden_size,num_of_hidden_layers,batch_first=True,bidirectional=bidirectional,dropout=dropout
                               if dropout!=0 and num_of_hidden_layers!=1 else 0)
        elif self.cell_type == "gru":
            self.cell = nn.GRU(embedding_size,hidden_size,num_of_hidden_layers,batch_first=True,bidirectional=bidirectional,dropout=dropout
                               if dropout!=0 and num_of_hidden_layers!=1 else 0)
        elif self.cell_type == "lstm":
            self.cell = nn.LSTM(embedding_size,hidden_size,num_of_hidden_layers,batch_first=True,bidirectional=bidirectional,dropout=dropout
                                if dropout!=0 and num_of_hidden_layers!=1 else 0)
        else:
            print("Not a valid cell type")
        if bidirectional==True:
            self.combine = nn.Linear(2*hidden_size,hidden_size)
        
        
    def forward(self,input,seq_len):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        N= input.shape[0]
        D = 2 if self.bidirectional else 1
        h0 = (torch.rand(D*self.num_of_hidden_layers,N,self.hidden_size) * (2/(self.hidden_size)**0.5) - (1/(self.hidden_size)**0.5)).to(device=device)
        c0 = (torch.rand(D*self.num_of_hidden_layers,N,self.hidden_size) * (2/(self.hidden_size)**0.5) - (1/(self.hidden_size)**0.5)).to(device=device)
        self.h0 = nn.parameter.Parameter(h0)
        self.c0 = nn.parameter.Parameter(c0)
        self.embedded_x=self.embedding(input)
        self.embedded_x = self.dropout_layer(self.embedded_x)

        self.packed_emb_x = pack_padded_sequence(self.embedded_x,seq_len.to(device="cpu"),batch_first=True,enforce_sorted=False)  
        if self.cell_type=="lstm":
        
            self.packed_output,(self.hidden_last,self.cell_last)= self.cell(self.packed_emb_x,(self.h0,self.c0))
            self.output= pad_packed_sequence(self.packed_output,batch_first=True)[0]
            self.output = self.combine(self.output)
            if self.bidirectional:
                out1 = self.combine(torch.cat((self.hidden_last[:self.num_of_hidden_layers],self.hidden_last[self.num_of_hidden_layers:]),dim=2))
                out2 = self.combine(torch.cat((self.cell_last[:self.num_of_hidden_layers],self.cell_last[self.num_of_hidden_layers:]),dim=2))


            else:
                out1 = self.hidden_last
                out2 = self.cell_last
        else:
            self.packed_output,self.hidden_last = self.cell(self.packed_emb_x,self.h0)
            self.output= pad_packed_sequence(self.packed_output,batch_first=True)[0]
            self.output = self.combine(self.output)
            if self.bidirectional:
                out = self.combine(torch.cat((self.hidden_last[:self.num_of_hidden_layers],self.hidden_last[self.num_of_hidden_layers:]),dim=2))
              
            else:
                out = self.hidden_last
        
        return (out1,out2) if self.cell_type=="lstm" else out , self.output
    
class decoder(nn.Module):
    def __init__(self,hidden_size,num_of_hidden_layers,dict_size,embedding_size,cell_type="rnn",dropout=0):
        super(decoder,self).__init__()
        self.hidden_size=hidden_size
        self.num_of_hidden_layers=num_of_hidden_layers
        self.dict_size=dict_size
        self.embedding_size=embedding_size
        self.cell_type=cell_type
        self.attn_mech = False
        self.dropout=dropout
        self.dropout_layer = None

        self.embedded_y=None
        self.hidden_output=None
        self.hidden_last=None
        self.cell_last=None
        self.linoutput=None
        
        self.embedding = nn.Embedding(dict_size,embedding_size)
        self.dropout_layer = nn.Dropout(self.dropout)

        if self.cell_type == "rnn":
            self.cell = nn.RNN(embedding_size,hidden_size,num_of_hidden_layers,batch_first=True,dropout=dropout if dropout!=0 and num_of_hidden_layers!=1 else 0)
        elif self.cell_type == "gru":
            self.cell = nn.GRU(embedding_size,hidden_size,num_of_hidden_layers,batch_first=True,dropout=dropout if dropout!=0 and num_of_hidden_layers!=1 else 0)
        elif self.cell_type == "lstm":
            self.cell = nn.LSTM(embedding_size,hidden_size,num_of_hidden_layers,batch_first=True,dropout=dropout if dropout!=0 and num_of_hidden_layers!=1 else 0)
        else:
            print("Not a valid cell type")
        
        self.lin = nn.Linear(hidden_size,dict_size)


    def forward(self,input,h0,c0):

        self.embedded_y=self.embedding(input)
        self.embedded_y = self.dropout_layer(self.embedded_y)
        if self.cell_type=="lstm":
            self.hidden_output,(self.hidden_last,self.cell_last)=self.cell(self.embedded_y,(h0,c0))
  
        else:
            self.hidden_output,self.hidden_last=self.cell(self.embedded_y,h0)

        self.lin_output= self.lin(self.hidden_output)

        
        return self.lin_output, (self.hidden_last,self.cell_last) if self.cell_type=="lstm" else self.hidden_last
    
class attention_decoder(nn.Module):
    def __init__(self,hidden_size,num_of_hidden_layers,dict_size,embedding_size,cell_type,encoder_hidden_size,dropout=0):
        super(attention_decoder,self).__init__()
        self.hidden_size=hidden_size
        self.num_of_hidden_layers = num_of_hidden_layers
        self.dict_size=dict_size
        self.embedding_size=embedding_size
        self.cell_type=cell_type
        self.attn_mech=True
        self.dropout=dropout
        self.dropout_layer = None

        self.embedded_y=None
        self.attention=None
        self.attn_weights=None
        self.weighted_sum = None
        self.combined_input=None
        self.hidden_output=None
        self.hidden_last=None
        self.cell_last=None
        self.lin_output=None
      

        self.embedding = nn.Embedding(dict_size,embedding_size)
        self.dropout_layer = nn.Dropout(self.dropout)


        self.Watt = nn.Linear(encoder_hidden_size,hidden_size)
        self.Uatt = nn. Linear(hidden_size,hidden_size)
        self.Vatt = nn.Parameter( torch.rand(hidden_size)*(2/(hidden_size)**0.5) - (1/(hidden_size)**0.5))

        self.combine = nn.Linear(embedding_size+encoder_hidden_size,embedding_size)
        
        if self.cell_type == "rnn":
            self.cell = nn.RNN(embedding_size,hidden_size,self.num_of_hidden_layers,batch_first=True,dropout=dropout 
                               if dropout!=0 and num_of_hidden_layers!=1 else 0)
        elif self.cell_type == "gru":
            self.cell = nn.GRU(embedding_size,hidden_size,self.num_of_hidden_layers,batch_first=True,dropout=dropout
                               if dropout!=0 and num_of_hidden_layers!=1 else 0)
        elif self.cell_type == "lstm":
            self.cell = nn.LSTM(embedding_size,hidden_size,self.num_of_hidden_layers,batch_first=True,dropout=dropout 
                                if dropout!=0 and num_of_hidden_layers!=1 else 0)
        else:
            print("Not a valid cell type")

        self.attn_layer = nn.Softmax(dim=1)

        self.lin = nn.Linear(hidden_size,dict_size)
  

    def forward(self,input,h0,c0,encoder_hidden_states,attn_mask):
        
        
        self.embedded_y = self.embedding(input)

        self.attention = torch.tanh((self.Watt(torch.permute(encoder_hidden_states,[1,0,2])) + self. Uatt(h0[-1]))) @ self.Vatt

        self.attn_weights = torch.permute(self.attn_layer(self.attention),[1,0]) * attn_mask
        
        self.weighted_sum = (self.attn_weights.unsqueeze(1) @ encoder_hidden_states)
    
        self.combined_input = self.combine(torch.cat((self.embedded_y,self.weighted_sum),dim=2))

        self.combined_input = self.dropout_layer(self.combined_input)

        if self.cell_type=="lstm":
            self.hidden_output,(self.hidden_last,self.cell_last)=self.cell(self.combined_input,(h0,c0))

        else:
            self.hidden_output,self.hidden_last=self.cell(self.combined_input,h0)
        
        self.lin_output= self.lin(self.hidden_output)
        
        return self.lin_output, (self.hidden_last,self.cell_last) if self.cell_type=="lstm" else self.hidden_last
