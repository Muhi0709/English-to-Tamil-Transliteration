import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from encoder_decoder import encoder,decoder,attention_decoder
from data_loader_vocab_builder import *
from queue import Queue
import wandb

wandb.login()






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

(latin_script,latin_script_idx2word,dev_script_word2idx,dev_script_idx2word),(
    train_data,max_length_x,max_length_y),(val_data,max_length_val_x,max_length_val_y),(
    test_data,max_length_test_x,max_length_test_y)=load_data()

train_data = train_data.to(device=device)
val_data = val_data.to(device=device)
test_data = test_data.to(device=device)

default_config={"cell_type":"rnn","hidden_size":256,"enc_embedding_size":256,"enc_dict_size": len(latin_script_idx2word),
                       "num_hl":4,"bidirectional":True,"dec_embedding_size":128,
                       "dec_dict_size": len(dev_script_word2idx),"loss_func":"cross_entropy","optimizer":"Adam",
                        "max_epoch":30,"batch_size":128,"learning_rate":0.002,
                        "beta_1":0.99,"beta_2":0.999,"epsilon":10**-8,"wei_decay":0.005,
                        "early_stopping":True,"patience":4,"wandb_log":False,"teacher_forcing":0.2,"attn_mech":True,"dropout":0.5,
                        "beam_search":0}


def do_early_stopping(patience,prev_loss,curr_loss,inc_val_count,enc_model_history,dec_model_history,enc,dec):
    if curr_loss>prev_loss:
        inc_val_count[0]+=1
    else:
        inc_val_count[0]=0
            
    if inc_val_count[0] == patience:
        #set model
        esd=enc_model_history.get()
        enc.load_state_dict(esd)
        dsd = dec_model_history.get()
        dec.load_state_dict(dsd)
        return True
    return False

def update_history(patience,enc_model_history,dec_model_history,enc,dec):
    with torch.no_grad():
        if enc_model_history.qsize()<patience+1:
            enc_model_history.put(enc.state_dict())
            dec_model_history.put(dec.state_dict())
        else:
            enc_model_history.get()
            enc_model_history.put(enc.state_dict())
            dec_model_history.get()
            dec_model_history.put(dec.state_dict())

def compute_loss_accuracy(x,y,max_x,max_y,enc1,dec1,seq_len_x,seq_len_y,testing=False,beam_search=0,crit=None):
    N=len(x)
    predicted_y=torch.zeros(y.shape[0],y.shape[1]-1 if not testing else 30).to(device=device)
    decoder_c0=None
    decoder_h0=None
    loss=torch.tensor(0.0).to(device=device)
    accuracy=0.0

    enc1.eval()
    dec1.eval()

    with torch.no_grad():
        if testing and beam_search:
              #do beam_search
            
            for i in range(N):
                if dec1.attn_mech:
                    attn_mask= torch.empty(1 ,max_x)
                    for i in range(1):
                        a= [1.0] * (seq_len_x[i]) + [0.0]*(max_x-seq_len_x[i])
                        attn_mask[i]=torch.tensor(a)
                a_prev = decoder_h0[:,i,:].unsqueeze(1),decoder_c0[:,i,:].unsqueeze(1)

                inputs= Queue()
                inputs.put((torch.tensor([[dev_script_word2idx["<sow>"]]]),a_prev,0.0))
                potential_candidates=[]
                for j in range(30):
                    while not inputs.empty():
                        weight = len(inputs)
                        input,a_prev,_ = inputs.get() [:,-1,:]
                        if not dec1.attn_mech:
                            probabilities,a= dec1(input,a_prev[0],a_prev[1])
                        else:
                            probabilities,a = dec1(input,a_prev[0],a_prev[1],encoder_hidden,attn_mask)

                        scores , candidates = torch.sort(probabilities,dim=2)
                        scores = scores [:,:,-beam_search:]
                        candidates = candidates[:,:,-beam_search]
                        if j==0:
                            for k in range(beam_search):
                                potential_candidates.append( (candidates[:,:,k],a,scores[:,:,k].item()))
                        else:
                            for k in range(beam_search):
                                potential_candidates.append( (torch.cat((input,candidates[:,:,k]),dim=1),a,scores[:,:,k].item()))
                    
                    potential_candidates.sort(desc=True,key = lambda x: x[2])
                    for k in range(beam_search):
                        inputs.put(potential_candidates[k])
                
                flag=0
                for pot_seq,_,_ in potential_candidates:
                    if torch.all(y[i,1:seq_len_y[i]]== pot_seq[i,:seq_len_y[i]-1]):
                        predicted_y[i:,:seq_len_y[i]-1] = pot_seq[i,:seq_len_y[i]-1]
                        accuracy+=1
                        flag=1
                        break
                if flag==0:
                    predicted_y[i]=potential_candidates[0][0]
            accuracy=accuracy*100/N
          
        else:
            if N>5120:
                n=0
                batch_size=int(N/10)
            else:
                n=0
                batch_size = N
            while n<N:
                if enc1.cell_type=="lstm":
                    (decoder_h0,decoder_c0),encoder_hidden = enc1(x[n:n+batch_size],seq_len_x[n:n+batch_size])
                else:
                    decoder_h0,encoder_hidden = enc1(x[n:n+batch_size],seq_len_x[n:n+batch_size])

                input= torch.tensor([[dev_script_word2idx["<sow>"]]]*batch_size).to(device=device)
                if dec1.attn_mech:
                    attn_mask= torch.sign( x[n:n+batch_size,:max(seq_len_x[n:n+batch_size])])
        
                
                for i in range(max_y -1 if not testing else 30):
                    if not dec1.attn_mech:
                        probabilities,a= dec1(input,decoder_h0,decoder_c0)
                    else:
                        probabilities,a = dec1(input,decoder_h0,decoder_c0,encoder_hidden,attn_mask)
                        
                    _,predicted_words = torch.max(probabilities,dim=2)
                    loss += crit ( torch.permute(probabilities,[0,2,1]), torch.unsqueeze(y[n:n+batch_size,i+1],1))
                    predicted_y[n:n+batch_size,i] = torch.flatten(predicted_words)
                    input = predicted_words
                    if dec1.cell_type=="lstm":
                        decoder_h0 = a[0]
                        decoder_c0=a[1]
                    else:
                        decoder_h0 =a
                n+=batch_size

            for i in range(N):
                if  torch.all(y[i,1:seq_len_y[i]] == predicted_y[i,:seq_len_y[i]-1]):
                    accuracy+=1
            accuracy= accuracy*100/N
            loss/=N
                
    return (loss.item(),accuracy,predicted_y) if not testing else (accuracy,predicted_y)

def predictions(enc,dec,test_data,dev_dict_idx2word,dev_dict_word2idx,ignore_accuracy = False,beam_search=0):

    N=len(test_data)
    predicted_words_list=[]
    actual_words_list=[]

    if not ignore_accuracy:
        acc,predicted_words = compute_loss_accuracy(test_data[:,:max_length_test_x],test_data[:,max_length_test_x:-2],max_length_test_y,enc,dec,
                                                         test_data[:,:-2],test_data[:,:-1],True,beam_search)
    else:
        _,predicted_words = compute_loss_accuracy(test_data[:,:max_length_test_x],test_data[:,max_length_test_x:-2],max_length_test_y,enc,dec,
                                                    test_data[:,:-2],test_data[:,:-1],True,beam_search)
    
    actual_words = test_data[:,max_length_test_x:-2]
    for i in range(N):
        a=""
        for idx in predicted_words[i]:
            if idx == dev_dict_word2idx["<eow>"]:
                break
            a+=dev_dict_idx2word[idx]
        predicted_words_list.append(a)
    for i in range(N):
        a=""
        for idx in actual_words[i]:
            if idx== dev_dict_word2idx["<sow>"]:
                continue
            elif idx == dev_dict_word2idx["<eow>"]:
                break
            a+=dev_dict_idx2word[idx]
        actual_words_list.append(a)


    actual_words_list = np.array(actual_words_list)
    predicted_words_list = np.array(predicted_words_list)
    result=np.hstack((actual_words_list,predicted_words_list)).T
    
    if not dec.attn_mech:
        np.savetxt("predictions_vanilla.csv",result,delimiter=",")
    else:
        np.savetxt("predictions_attention.csv",result,delimiter=",")
        
    return (predicted_words_list if ignore_accuracy else predicted_words_list,acc)


def attention_heatmap(enc,dec,test_data,max_length_test_x,dev_script_word2idx,dev_script_idx2words,
                      latin_script_idx2word):
    N=len(test_data)
    permute = torch.randperm(N)
    
    test_data_sampled =  (test_data[permute])[:10]
    test_data_x = test_data_sampled[:,: max_length_test_x]
    test_data_xseq_lens = test_data_sampled[:,-2]


    for i in range(10):
        enc.eval()
        dec.eval()
        attention_weight_matrix=torch.zeros(test_data_xseq_lens[i],30)
        prediction_list = []
    
        with torch.no_grad():
            if enc.cell_type=="lstm":
                (decoder_h0,decoder_c0),encoder_hidden = enc(test_data_x[i].unsqueeze(0),test_data_xseq_lens[i].unsqueeze(0))
            else:
                decoder_h0,encoder_hidden = enc(test_data_x[i].unsqueeze(0), test_data_xseq_lens[i].unsqueeze(0))

            attn_mask= torch.empty(1,max_length_test_x)
            a= [1.0] * (test_data_xseq_lens[i]) + [0.0]*(max_length_test_x-test_data_xseq_lens[i])
            attn_mask[i]=torch.tensor(a)

            input= torch.tensor([[dev_script_word2idx["<sow>"]]])
            for j in range(30):

                probabilities,a = dec(input,decoder_h0,decoder_c0,encoder_hidden,attn_mask)
                _,predicted = torch.max(probabilities,dim=2)

                attention_weight_matrix[:,j] = dec.attn_weights[:test_data_xseq_lens[i]]
                prediction_list.append(dev_script_idx2words[predicted[0,0]])

                if predicted.item() == dev_script_word2idx["eow"]:
                    break
                input = torch.tensor(predicted)
                if dec.cell_type=="lstm":
                    decoder_h0 = a[0]
                    decoder_c0=a[1]
                else:
                    decoder_h0 =a
        
        attention_weight_matrix = attention_weight_matrix[:,:j]
        x_list = [latin_script_idx2word[idx] for idx in test_data_x[i,:test_data_xseq_lens[i]]]

        #heatmap_plot


enc = encoder(hidden_size=default_config["hidden_size"],num_of_hidden_layers=default_config["num_hl"],dict_size=default_config["enc_dict_size"],
               embedding_size=default_config["enc_dict_size"],cell_type=default_config["cell_type"],
               bidirectional=default_config["bidirectional"],dropout=default_config["dropout"])
    
enc.to(device=device)
    
if not default_config["attn_mech"]:
    dec = decoder(hidden_size= default_config["hidden_size"],
        num_of_hidden_layers=default_config["num_hl"],dict_size=default_config["dec_dict_size"],
        embedding_size=default_config["dec_embedding_size"],
        cell_type=default_config["cell_type"],dropout= default_config["dropout"])
    
else:
    dec = attention_decoder(hidden_size=default_config["hidden_size"],
                            num_of_hidden_layers=default_config["num_hl"],dict_size=default_config["dec_dict_size"],
                            embedding_size=default_config["dec_embedding_size"],
                            cell_type=default_config["cell_type"],
                            encoder_hidden_size=default_config["hidden_size"],dropout= default_config["dropout"])

dec.to(device=device)


def training(enc,dec,dev_script_word2idx,train_data,val_data,max_length_x,max_length_y,max_length_val_x,max_length_val_y,
             loss_func=default_config["loss_func"],
             optimizer=default_config["optimizer"],
             max_epoch=default_config["max_epoch"],batch_size=default_config["batch_size"],learning_rate=default_config["learning_rate"],
             beta_1=default_config["beta_1"],beta_2=default_config["beta_2"],epsilon=default_config["epsilon"],
             wei_decay=default_config["wei_decay"],early_stopping=default_config["early_stopping"],
             patience=default_config["patience"],
             wandb_log=default_config["wandb_log"],teacher_forcing=default_config["teacher_forcing"],
             attn_mech=default_config["attn_mech"],beam_search= default_config["beam_search"]):
    
    
    enc_model_history= None
    dec_model_history=None
    incr_val_loss_count= None
    val_acc_hist= []
    prev_val_loss=float("inf")
  

    if early_stopping:
        enc_model_history= Queue()
        enc_model_history.put(enc.state_dict())
        dec_model_history = Queue()
        dec_model_history.put(enc.state_dict())
        incr_val_loss_count= [0]
        
    
    criterion=None
    enc_opt=None
    dec_opt=None

    if loss_func== "cross_entropy":
        criterion = nn.CrossEntropyLoss(ignore_index=0,reduction="sum")
    
    if optimizer=="Adam":
        enc_opt= optim.Adam(enc.parameters(),lr=learning_rate,betas=(beta_1,beta_2),eps=epsilon,weight_decay=wei_decay)
        dec_opt= optim.Adam(dec.parameters(),lr=learning_rate,betas=(beta_1,beta_2),eps=epsilon,weight_decay=wei_decay)

    elif optimizer== "NAdam":
        enc_opt= optim.NAdam(enc.parameters(),lr=learning_rate,betas=(beta_1,beta_2),eps=epsilon,weight_decay=wei_decay)
        dec_opt= optim.NAdam(dec.parameters(),lr=learning_rate,betas=(beta_1,beta_2),eps=epsilon,weight_decay=wei_decay)

    
    train_loader= DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)


    for epoch in range(1,max_epoch+1):
        for data in train_loader:
            enc_opt.zero_grad()
            dec_opt.zero_grad()

            batch_size=len(data)
            loss=torch.tensor(0.0).to(device=device)
            train_x = data[:,:max_length_x].to(device=device)
            train_y = data[:,max_length_x:-2].to(device=device)
            seq_lens_train_x= data[:,-2].to(device=device)
            seq_lens_train_y= data[:,-1].to(device=device)

            decoder_c0=None
            decoder_h0=None
            attn_mask=None

            if enc.cell_type=="lstm":

                (decoder_h0,decoder_c0),encoder_hidden = enc(train_x,seq_lens_train_x)

            else:
                decoder_h0,encoder_hidden = enc(train_x,seq_lens_train_x)

            
            if attn_mech:
                attn_mask= torch.sign(train_x[:,:max(seq_lens_train_x)])


            if epoch <= int(teacher_forcing * max_epoch):
                if not attn_mech:
                    p,_ = dec(train_y[:,:max(seq_lens_train_y)-1],decoder_h0,decoder_c0)
                    probabilities=torch.permute(p,[0,2,1])
                    
                    loss = criterion( probabilities, train_y[:,1:max(seq_lens_train_y)])
                else:
                    input= train_y[:,0].unsqueeze(1)
                    for i in range(max(seq_lens_train_y)-1):
                        probabilities,a = dec(input,decoder_h0,decoder_c0,encoder_hidden,attn_mask)
                        _,predicted_words = torch.max(probabilities,dim=2)
                        loss += criterion ( torch.permute(probabilities,[0,2,1]),train_y[:,i+1].unsqueeze(dim=1))
                        input = train_y[:,i+1].unsqueeze(1)
                        if dec.cell_type=="lstm":
                            decoder_h0 = a[0]
                            decoder_c0= a[1]
                        else:
                            decoder_h0 = a

            else:
        
                input= torch.tensor([[dev_script_word2idx["<sow>"]]]*batch_size).to(device=device)
                for i in range(max(seq_lens_train_y)-1):
                    if not attn_mech:
                        probabilities,a= dec(input,decoder_h0,decoder_c0)
                    else:
                        probabilities,a = dec(input,decoder_h0,decoder_c0,encoder_hidden,attn_mask)

                    _,predicted_words = torch.max(probabilities,dim=2)
                    loss += criterion ( torch.permute(probabilities,[0,2,1]), torch.unsqueeze(train_y[:,i+1],1))
                    input = predicted_words
                    if dec.cell_type=="lstm":
                        decoder_h0 = a[0]
                        decoder_c0=a[1]
                    else:
                        decoder_h0 =a

            loss.backward()


            
            enc_opt.step()
            dec_opt.step()

          

        train_loss,train_acc,_ = compute_loss_accuracy(train_data[:,:max_length_x],train_data[:,max_length_x:-2],max_length_x,
                                                       max_length_y,enc,dec,train_data[:,-2],train_data[:,-1],False,beam_search,criterion)
        val_loss,val_acc,_ = compute_loss_accuracy(val_data[:,:max_length_val_x],val_data[:,max_length_val_x:-2],max_length_val_x,max_length_val_y,enc,dec,
                                                 val_data[:,-2],val_data[:,-1],False,beam_search,criterion)
        

        val_acc_hist.append(val_acc)
        print("epoch: {}".format(epoch))
        print("=>train_acc={},train_loss={},val_acc={},val_loss={}".format(train_acc,train_loss,
                                                                                val_acc,val_loss))
        print("-"*200)
        val_acc_hist.append(val_acc)
        if wandb_log:
            wandb.log({"epoch":epoch,"train_accuracy":train_acc,
                        "train_loss":train_loss,
                        "val_accuracy":val_acc,
                        "val_loss": val_loss})
            
        if early_stopping:
            update_history(patience,enc_model_history,dec_model_history,enc,dec)
            if do_early_stopping(patience,prev_val_loss,val_loss,incr_val_loss_count,enc_model_history,dec_model_history,enc,dec):
                    print("Early stopping event has occured!!!")
                    print("val_accuracy (before the event): ", val_acc_hist[int(-1*(patience+1))])
                    break
            prev_val_loss=val_loss
        #print/logging

        enc.train()
        dec.train()
    
training(enc,dec,dev_script_word2idx,train_data,val_data,max_length_x,max_length_y,max_length_val_x,max_length_val_y)