import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_character_corpus(x,y,latin_script_word2idx,latin_script_idx2word,dev_script_idx2word,dev_script_word2idx):

    N=len(x)
    for i in range(N):
        for ele in x[i]:
            if ele not in latin_script_word2idx:
                latin_script_word2idx[ele]=len(latin_script_word2idx)
                latin_script_idx2word[len(latin_script_idx2word)]=ele


        for ele in y[i]:
            if ele not in dev_script_word2idx:
                dev_script_word2idx[ele]=len(dev_script_word2idx)
                dev_script_idx2word[len(dev_script_idx2word)]=ele


def words_to_indices(x,y,latin_script,dev_script_word2idx):
    N=len(x)
    max_length_x=-float("inf")
    max_length_y=-float("inf")
    x_list=[]
    y_list=[]
    seq_lens_x=[]
    seq_lens_y=[]
    for i in range(N):
        a=[]
        for ele in x[i]:
            if ele in latin_script:
                a.append(latin_script[ele])
            else:
                a.append(latin_script["<un>"])
        a.append(latin_script["<eow>"])
        seq_lens_x.append(len(a))
        max_length_x=max(max_length_x,len(a))
        x_list.append(a)

        a=[]
        a.append(dev_script_word2idx["<sow>"])
        for ele in y[i]:
            if ele in dev_script_word2idx:
                a.append(dev_script_word2idx[ele])
            else:
                a.append(dev_script_word2idx["<un>"])
        a.append(dev_script_word2idx["<eow>"])
        seq_lens_y.append(len(a))
        max_length_y=max(max_length_y,len(a))
        y_list.append(a)

    for i in range(N):
        if len(x_list[i])<max_length_x:
            x_list[i]=x_list[i]+[0]*(max_length_x-len(x_list[i]))
        if len(y_list[i])< max_length_y:
            y_list[i] = y_list[i] + [0]*(max_length_y-len(y_list[i]))
    return torch.cat((torch.tensor(x_list),torch.tensor(y_list),torch.tensor(
        seq_lens_x).unsqueeze(1),torch.tensor(seq_lens_y).unsqueeze(1)),dim=1),max_length_x,max_length_y

def load_data():
    train_pairs=np.genfromtxt("C:/Users/muhilan/Downloads/aksharantar_sampled/aksharantar_sampled/tam/tam_train.csv",
                          delimiter=",",encoding="utf-8",dtype=str)
    
    val_pairs=np.genfromtxt("C:/Users/muhilan/Downloads/aksharantar_sampled/aksharantar_sampled/tam/tam_valid.csv",
                          delimiter=",",encoding="utf-8",dtype=str)
    test_pairs=np.genfromtxt("C:/Users/muhilan/Downloads/aksharantar_sampled/aksharantar_sampled/tam/tam_test.csv",
                          delimiter=",",encoding="utf-8",dtype=str)
    latin_script_word2idx={"<pad>":0,"<un>":1,"<eow>":2}
    latin_script_idx2word={0:"<pad>",1:"<un>",2:"<eow>"}
    dev_script_word2idx={"<pad>":0,"<sow>":1,"<eow>":2,"<un>":3}
    dev_script_idx2word={0:"pad",1:"<sow>",2:"<eow>",3:"<un>"}


    build_character_corpus(train_pairs[:,0],train_pairs[:,1],latin_script_word2idx,latin_script_idx2word,dev_script_idx2word,dev_script_word2idx)
    train_data,max_length_x,max_length_y=words_to_indices(train_pairs[:,0],train_pairs[:,1],latin_script_word2idx,dev_script_word2idx)
    val_data,max_length_val_x,max_length_val_y=words_to_indices(val_pairs[:,0],val_pairs[:,1],latin_script_word2idx,dev_script_word2idx)
    test_data,max_length_test_x,max_length_test_y=words_to_indices(test_pairs[:,0],test_pairs[:,1],latin_script_word2idx,dev_script_word2idx)


    return (latin_script_word2idx,latin_script_idx2word,dev_script_word2idx,dev_script_idx2word),(
        train_data,max_length_x,max_length_y),(val_data,max_length_val_x,max_length_val_y),(test_data,
        max_length_test_x,max_length_test_y)