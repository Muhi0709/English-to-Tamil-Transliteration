import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #setting device to cuda if it is available

#build the dictionary for inputs(latin_script) and outputs("Tamil" script)
#character to index & index to character
def build_character_corpus(x,y,latin_script_word2idx,latin_script_idx2word,tam_script_idx2word,tam_script_word2idx):

    N=len(x)
    for i in range(N):
        for ele in x[i]:
            if ele not in latin_script_word2idx:
                latin_script_word2idx[ele]=len(latin_script_word2idx)
                latin_script_idx2word[len(latin_script_idx2word)]=ele


        for ele in y[i]:
            if ele not in tam_script_word2idx:
                tam_script_word2idx[ele]=len(tam_script_word2idx)
                tam_script_idx2word[len(tam_script_idx2word)]=ele

#load data and convert letters to indices using the built corpus/script dictionary
def words_to_indices(x,y,latin_script,tam_script_word2idx):
    N=len(x)
    max_length_x=-float("inf")
    max_length_y=-float("inf")
    x_list=[]
    y_list=[]
    seq_lens_x=[]
    seq_lens_y=[]
    for i in range(N):         
        a=[]
        for ele in x[i]:               #go over each character in the input(latin_script) word
            if ele in latin_script:
                a.append(latin_script[ele])    
            else:
                a.append(latin_script["<un>"])    #unallocated token for previously unseen character
        a.append(latin_script["<eow>"])            # end of word token to mark the end of the input word
        seq_lens_x.append(len(a))
        max_length_x=max(max_length_x,len(a))
        x_list.append(a)

        a=[]
        a.append(tam_script_word2idx["<sow>"])            # append start of word token to mark the start of the prediction/output word
        for ele in y[i]:                              #go over each character in the input(latin_script) word
            if ele in tam_script_word2idx:
                a.append(tam_script_word2idx[ele])
            else:
                a.append(tam_script_word2idx["<un>"])   #unallocated token for previously unseen character
        a.append(tam_script_word2idx["<eow>"])             # end of word token to mark the end of the prediction/output word
        seq_lens_y.append(len(a))
        max_length_y=max(max_length_y,len(a))
        y_list.append(a)

    for i in range(N):                                      #find the maximum of sequence lengths to do padding with pad token(index:0)
        if len(x_list[i])<max_length_x:
            x_list[i]=x_list[i]+[0]*(max_length_x-len(x_list[i]))
        if len(y_list[i])< max_length_y:
            y_list[i] = y_list[i] + [0]*(max_length_y-len(y_list[i]))
    return torch.cat((torch.tensor(x_list),torch.tensor(y_list),torch.tensor(
        seq_lens_x).unsqueeze(1),torch.tensor(seq_lens_y).unsqueeze(1)),dim=1),max_length_x,max_length_y

def load_data():   #load train,valid,test_pairs
    train_pairs=np.genfromtxt("aksharantar_sampled/tam/tam_train.csv",
                          delimiter=",",encoding="utf-8",dtype=str)
    
    val_pairs=np.genfromtxt("aksharantar_sampled/tam/tam_valid.csv",
                          delimiter=",",encoding="utf-8",dtype=str)
    test_pairs=np.genfromtxt("aksharantar_sampled/tam/tam_test.csv",
                          delimiter=",",encoding="utf-8",dtype=str)
    latin_script_word2idx={"<pad>":0,"<un>":1,"<eow>":2}
    latin_script_idx2word={0:"<pad>",1:"<un>",2:"<eow>"}
    tam_script_word2idx={"<pad>":0,"<sow>":1,"<eow>":2,"<un>":3}
    tam_script_idx2word={0:"pad",1:"<sow>",2:"<eow>",3:"<un>"}

    #build corpus
    build_character_corpus(train_pairs[:,0],train_pairs[:,1],latin_script_word2idx,latin_script_idx2word,tam_script_idx2word,tam_script_word2idx)
    #character to indices mapping becomes the new data
    train_data,max_length_x,max_length_y=words_to_indices(train_pairs[:,0],train_pairs[:,1],latin_script_word2idx,tam_script_word2idx)
    val_data,max_length_val_x,max_length_val_y=words_to_indices(val_pairs[:,0],val_pairs[:,1],latin_script_word2idx,tam_script_word2idx)
    test_data,max_length_test_x,max_length_test_y=words_to_indices(test_pairs[:,0],test_pairs[:,1],latin_script_word2idx,tam_script_word2idx)


    return (latin_script_word2idx,latin_script_idx2word,tam_script_word2idx,tam_script_idx2word),(
        train_data,max_length_x,max_length_y),(val_data,max_length_val_x,max_length_val_y),(test_data,
        max_length_test_x,max_length_test_y)
#return corpus,train_data(character indices),test_data(character indices),valid_data(character indices)