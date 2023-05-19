import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence

# N:batch_size
# L:sequence_length

class encoder(nn.Module):     #encoder network definition
    def __init__(self,hidden_size,num_of_hidden_layers,dict_size,embedding_size,cell_type="rnn",bidirectional=False,dropout=0):
        super(encoder,self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        D = 2 if bidirectional else 1

        self.hidden_size=hidden_size   #size of hidden layer
        self.num_of_hidden_layers=num_of_hidden_layers      #number of hl
        self.dict_size=dict_size                         #size of script dictionary
        self.embedding_size=embedding_size               #character embedding size(incidices to vector of give embedding size)
        self.cell_type=cell_type                         #cell type: rnn,lstm,gru
        self.bidirectional= bidirectional               #enable bidrectionality
        self.dropout=dropout                            #dropout layer probability


        self.dropout_layer=None
        self.h0=None
        self.c0=None
        self.embedded_x=None
        self.hidden_output = None
        self.hidden_last = None
        self.cell_last = None
        self.cell=None
        self.output=None
        # initial h0 and c0 of the encoder are treated as parameters of the model
        h0 = (torch.rand(D*self.num_of_hidden_layers,self.hidden_size) * (2/(self.hidden_size)**0.5) - (1/(self.hidden_size)**0.5)).to(device=device)
        c0 = (torch.rand(D*self.num_of_hidden_layers,self.hidden_size) * (2/(self.hidden_size)**0.5) - (1/(self.hidden_size)**0.5)).to(device=device)
        self.h0 = nn.parameter.Parameter(h0)
        self.c0 = nn.parameter.Parameter(c0)

        self.embedding = nn.Embedding(dict_size,embedding_size)   #embedding layer, input: N x L -> output : N x L x embedding size
        self.dropout_layer = nn.Dropout(self.dropout)            #dropout layer with given probability

        if self.cell_type == "rnn":  # cell layer: input: Nx Lx embedding size -> output: N X L X hidden size(2*hidden size if bidirection =True)
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
        if bidirectional==True:             #if birectional, use a linear layer to combine forward & reverse hidden states, instead of concantenation
            self.combine = nn.Linear(2*hidden_size,hidden_size)
        
        
    def forward(self,input,seq_len):
        N= input.shape[0]
        h0_conc=torch.cat((self.h0.unsqueeze(1),)*N,dim=1)
        c0_conc=torch.cat((self.c0.unsqueeze(1),)*N,dim=1)
        self.embedded_x=self.embedding(input)
        self.embedded_x = self.dropout_layer(self.embedded_x)  #incides->embedding->dropout

        self.packed_emb_x = pack_padded_sequence(self.embedded_x,seq_len.to(device="cpu"),batch_first=True,enforce_sorted=False) 
        #pack padded sequence to ignore pad index(0) durinf cell layer computation (pad doesnot afectthe final hidden states from the encoder) 
        if self.cell_type=="lstm":
            self.packed_output,(self.hidden_last,self.cell_last)= self.cell(self.packed_emb_x,(h0_conc,c0_conc))
            self.output= pad_packed_sequence(self.packed_output,batch_first=True)[0] #unpack the last layer hidden states for the entire sequence length L

            if self.bidirectional:
                self.output = self.combine(self.output)
                out1 = self.combine(torch.cat((self.hidden_last[:self.num_of_hidden_layers],self.hidden_last[self.num_of_hidden_layers:]),dim=2))
                out2 = self.combine(torch.cat((self.cell_last[:self.num_of_hidden_layers],self.cell_last[self.num_of_hidden_layers:]),dim=2))
        # if cell type is lstm: output both final(last sequence) hidden state h0 & cell state c0 else output only final hidden state h0 

            else:
                out1 = self.hidden_last
                out2 = self.cell_last
        else:
            self.packed_output,self.hidden_last = self.cell(self.packed_emb_x,h0_conc)
            self.output= pad_packed_sequence(self.packed_output,batch_first=True)[0] #unpack the last layer hidden states for the entire sequence length L

            if self.bidirectional:
                self.output = self.combine(self.output)
                out = self.combine(torch.cat((self.hidden_last[:self.num_of_hidden_layers],self.hidden_last[self.num_of_hidden_layers:]),dim=2))
              
            else:
                out = self.hidden_last

        return (out1,out2) if self.cell_type=="lstm" else out , self.output
    
class decoder(nn.Module):   # decoder network definition (without attention mechanism)
    def __init__(self,hidden_size,num_of_hidden_layers,dict_size,embedding_size,cell_type="rnn",dropout=0):
        super(decoder,self).__init__()
        self.hidden_size=hidden_size    #size of hidden layer
        self.num_of_hidden_layers=num_of_hidden_layers  #number of hl
        self.dict_size=dict_size        #size of script dictionary
        self.embedding_size=embedding_size #character embedding size(incidices to vector of give embedding size)
        self.cell_type=cell_type           #cell type: rnn,lstm,gru
        self.attn_mech = False             #attention set to False
        self.dropout=dropout               #dropout probability
        self.dropout_layer = None

        self.embedded_y=None
        self.hidden_output=None
        self.hidden_last=None
        self.cell_last=None
        self.linoutput=None
        
        self.embedding = nn.Embedding(dict_size,embedding_size) #embedding layer, input: N x L -> output : N x L x embedding size
        self.dropout_layer = nn.Dropout(self.dropout) #dropout layer with given probability

        if self.cell_type == "rnn": #cell layer: input: Nx Lx embedding size -> output: N X L X hidden size
            self.cell = nn.RNN(embedding_size,hidden_size,num_of_hidden_layers,batch_first=True,dropout=dropout if dropout!=0 and num_of_hidden_layers!=1 else 0)
        elif self.cell_type == "gru":
            self.cell = nn.GRU(embedding_size,hidden_size,num_of_hidden_layers,batch_first=True,dropout=dropout if dropout!=0 and num_of_hidden_layers!=1 else 0)
        elif self.cell_type == "lstm":
            self.cell = nn.LSTM(embedding_size,hidden_size,num_of_hidden_layers,batch_first=True,dropout=dropout if dropout!=0 and num_of_hidden_layers!=1 else 0)
        else:
            print("Not a valid cell type")
        
        self.lin = nn.Linear(hidden_size,dict_size) #linear layer to convert hidden states to output (vector of dimension equal to devanagari script dict size)
        # N X L X hidden size -> N X L X dict_size (like a classification problem)
        # Note no softmax, as nn.CrossEntropy is used(directly takes logits)

    def forward(self,input,h0,c0):

        self.embedded_y=self.embedding(input)
        self.embedded_y = self.dropout_layer(self.embedded_y)  #incides->embedding->dropout
        if self.cell_type=="lstm":
            self.hidden_output,(self.hidden_last,self.cell_last)=self.cell(self.embedded_y,(h0,c0))
       # if cell type is lstm: output both final hidden state h0 & cell state c0 else output only final hidden state h0 
        else:
            self.hidden_output,self.hidden_last=self.cell(self.embedded_y,h0)

        self.lin_output= self.lin(self.hidden_output)     #hidden state to classes (character indices:logits)

        
        return self.lin_output, (self.hidden_last,self.cell_last) if self.cell_type=="lstm" else self.hidden_last
    
class attention_decoder(nn.Module):  # decoder network definition (with attention mechanism)
    def __init__(self,hidden_size,num_of_hidden_layers,dict_size,embedding_size,cell_type,encoder_hidden_size,dropout=0):
        super(attention_decoder,self).__init__()
        self.hidden_size=hidden_size         #size of hidden layer
        self.num_of_hidden_layers = num_of_hidden_layers  #number of hl
        self.dict_size=dict_size                #size of script dictionary
        self.embedding_size=embedding_size         #character embedding size(incidices to vector of give embedding size)
        self.cell_type=cell_type                   #cell type: rnn,lstm,gru
        self.attn_mech=True                      #atten mech set to true
        self.dropout=dropout                   #dropout probability
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
      

        self.embedding = nn.Embedding(dict_size,embedding_size)  #embedding layer, input: N x L -> output : N x L x embedding size
        self.dropout_layer = nn.Dropout(self.dropout)    #dropout layer with given probability

        #implements Vatt.(Watt @ h_j +Uatt @ s_t-1)
        # only uses the last layer of encoder hidden states  and last layer of s_(t-1)
        self.Watt = nn.Linear(encoder_hidden_size,hidden_size) # attention W matrix/layer: L(=encoder/input max seq length) X N XH(encoder hidden size)-> L(==encoder/input max seq length) X N XHhidden size)
        self.Uatt = nn. Linear(hidden_size,hidden_size) # attention U matrix/layer: L(=1)x N x H(hidden size)->L(=1)x N x H(hidden size)
        self.Vatt = nn.Parameter( torch.rand(hidden_size)*(2/(hidden_size)**0.5) - (1/(hidden_size)**0.5)) #dot product with Vatt to get scaler attention weights(e_jt)
        
        # linear layer to combine the weighted encoder hidden states & embedded input(to size equal to embedding size),instead of concantenation
        self.combine = nn.Linear(embedding_size+encoder_hidden_size,embedding_size)  
        
        if self.cell_type == "rnn":   #cell layer: input: Nx Lx embedding size -> output: N X L X hidden size
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

        self.attn_layer = nn.Softmax(dim=1)      # attention layer to convert e_jt's to probabilities(attn weights)

        self.lin = nn.Linear(hidden_size,dict_size)  #final linear layer: cell hidden layer to logits
        # N X L X hidden size -> N X L X dict_size (like a classification problem)
        # Note no softmax, as nn.CrossEntropy is used(directly takes logits)

    def forward(self,input,h0,c0,encoder_hidden_states,attn_mask):
        
        
        self.embedded_y = self.embedding(input) #incides->embedding

        self.attention = torch.tanh((self.Watt(torch.permute(encoder_hidden_states,[1,0,2])) + self. Uatt(h0[-1]))) @ self.Vatt #e_jt
        #e_jt -> weights(by softmax), attn_mask matrix to ensure encoder sequences upto <eow> is only used
        self.attn_weights = self.attn_layer(torch.permute(self.attention,[1,0])) * attn_mask 
        
        self.weighted_sum = (self.attn_weights.unsqueeze(1) @ encoder_hidden_states)  #weighted sum of encoder states
    
        self.combined_input = self.combine(torch.cat((self.embedded_y,self.weighted_sum),dim=2)) #combined input to decoder cell

        self.combined_input = self.dropout_layer(self.combined_input)

        if self.cell_type=="lstm":
            self.hidden_output,(self.hidden_last,self.cell_last)=self.cell(self.combined_input,(h0,c0))
        # if cell type is lstm: output both final hidden state h0 & cell state c0 else output only final hidden state h0 
        else:
            self.hidden_output,self.hidden_last=self.cell(self.combined_input,h0)
        
        self.lin_output= self.lin(self.hidden_output)   #decoder hidden state to classes (character indices:logits)

        
        return self.lin_output, (self.hidden_last,self.cell_last) if self.cell_type=="lstm" else self.hidden_last
