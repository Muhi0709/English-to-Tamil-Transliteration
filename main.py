import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from data_loader_vocab_builder import *
from queue import Queue
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #check for cuda
torch.manual_seed(42)      #global random seed

(latin_script,latin_script_idx2word,dev_script_word2idx,dev_script_idx2word),(_,_,_),(_,_,_),(_,_,_)=load_data()  
#load the dictionaries using data_loader_vocab_builder_script



# set the defau;ts for 'training" function definitions 
default_config={"cell_type":"rnn","hidden_size":256,"enc_embedding_size":256,"enc_dict_size": len(latin_script_idx2word),
                       "num_hl":4,"bidirectional":True,"dec_embedding_size":128,
                       "dec_dict_size": len(dev_script_word2idx),"loss_func":"cross_entropy","optimizer":"Adam",
                        "max_epoch":30,"batch_size":128,"learning_rate":0.002,
                        "beta_1":0.99,"beta_2":0.999,"epsilon":10**-8,"wei_decay":0.005,
                        "early_stopping":True,"patience":4,"wandb_log":False,"teacher_forcing":0.2,"attn_mech":True,"dropout":0.5}

#helper function to implement early stopping, if early stopping condition is met(val_loss increasing for a defined "patience" period)
def do_early_stopping(patience,prev_loss,curr_loss,inc_val_count,enc_model_history,dec_model_history,enc,dec):
    if curr_loss>prev_loss:
        inc_val_count[0]+=1
    else:                       #increase counter if val_loss is increasing (after an epoch),else reset
        inc_val_count[0]=0   
            
    if inc_val_count[0] == patience:
        #set model
        esd=enc_model_history.get()
        enc.load_state_dict(esd)          #load the before patience period model parameters, incase of an early stopping event
        dsd = dec_model_history.get()
        dec.load_state_dict(dsd)
        return True
    return False

#keep track of models for a period of "patience",so that the model can be loaded in case an early stopping event occurs
def update_history(patience,enc_model_history,dec_model_history,enc,dec):  
    with torch.no_grad():
        if enc_model_history.qsize()<patience+1:
            enc_model_history.put(enc.state_dict())
            dec_model_history.put(dec.state_dict())
        else:
            enc_model_history.get()               #storing model dicts in a Queue
            enc_model_history.put(enc.state_dict())
            dec_model_history.get()
            dec_model_history.put(dec.state_dict())

def encoder_forward_prop(enc,x,seq_lens_x):        #forward propagation for the input words/encoder model
    decoder_c0=None
    if enc.cell_type=="lstm":
        (decoder_h0,decoder_c0),encoder_hidden = enc(x,seq_lens_x)   #calling the encoder.forward() method

    else:
        decoder_h0,encoder_hidden = enc(x,seq_lens_x)

    return decoder_h0,decoder_c0,encoder_hidden       # output h_n,c_n(last sequence  hidden & cell state ), last layer  hidden states (all L)

    
def decoder_forward_prop(dec,x,y,seq_lens_x,seq_lens_y,dev_script_word2idx,criterion,decoder_h0,decoder_c0,encoder_hidden,
                         teacher_forcing_bool=False,testing=False): #forward propagation for the decoder model
    
    loss=torch.tensor(0.0).to(device=device)
    len_y = (max(seq_lens_y)-1) if not testing else 30    #limit of 30 during testing(to prevent infinite predictions(no preediction of eow),30 arrived at based on maximum train sequence length of 23or 24)
    predicted_y = torch.zeros(len(x),len_y).to(device=device) 
    attn_mask= torch.sign(x[:,:max(seq_lens_x)]) if dec.attn_mech else None  #mask to ensure only encoder states upto <eow> are considered

    if teacher_forcing_bool :                        # teacher forcing is True, feed true y as input to decoder
        if not dec.attn_mech:
            p,_ = dec(y[:,:len_y],decoder_h0,decoder_c0)   #when model is not using atten_mechanism, true y's are feeded in one go(all letters in one go)
            probabilities=torch.permute(p,[0,2,1])          # calls decoder.forward()
            loss = criterion( probabilities, y[:,1:len_y+1])   #obtained logits are used for loss computation(cross entropy defau;t)
        else:
            input= y[:,0].unsqueeze(1)   ##when model is not using atten_mechanism, true y's are feeded in one by one(one letter in a time step) 
            for i in range(len_y):
                probabilities,a = dec(input,decoder_h0,decoder_c0,encoder_hidden,attn_mask)
                _,predicted_words = torch.max(probabilities,dim=2)   #compute and add loss for each time step until max(seq_lengths)/eow is reached.Note batch processing is used.
                loss += criterion ( torch.permute(probabilities,[0,2,1]),y[:,i+1].unsqueeze(dim=1))
                input = y[:,i+1].unsqueeze(1)
                if dec.cell_type=="lstm":
                    decoder_h0 = a[0]
                    decoder_c0= a[1]
                else:                     #output hidden states from prev time step, fed as input to  next
                    decoder_h0 = a

    else:        #when teacher forcing not enabled, predicted words at previous time step are fed as input to next
        #processing happens one time step after another
        # this block runs for both cases:if decoder uses atten or not 
        input= torch.tensor([[dev_script_word2idx["<sow>"]]]*len(x)).to(device=device)
        for i in range(len_y):
            if not dec.attn_mech:
                probabilities,a= dec(input,decoder_h0,decoder_c0)
            else:
                probabilities,a = dec(input,decoder_h0,decoder_c0,encoder_hidden,attn_mask)
            _,predicted_words = torch.max(probabilities,dim=2)
            predicted_y[:,i] = torch.flatten(predicted_words)
            if not testing:
                loss += criterion ( torch.permute(probabilities,[0,2,1]), torch.unsqueeze(y[:,i+1],1))
            input = predicted_words
            if dec.cell_type=="lstm":
                decoder_h0 = a[0]
                decoder_c0=a[1]
            else:
                decoder_h0 =a
    

    #return loss and the predicted words(indices) for the given inputs
    return loss,predicted_y



#loss accuracy calculating function, used during training & validation phase. Can be invoked for testing/inference. Does batch wise computation
def compute_loss_accuracy(x,y,enc1,dec1,seq_len_x,seq_len_y,dev_script_word2idx,testing=False,
                          crit=nn.CrossEntropyLoss(ignore_index=0,reduction="sum")): 
    N=len(x)
    predicted_y=torch.zeros(y.shape[0],y.shape[1]-1 if not testing else 30).to(device=device) #predicted words(indices) tensor for the given batch
    decoder_c0=None
    decoder_h0=None
    loss=torch.tensor(0.0).to(device=device)
    accuracy=0.0

    enc1.eval()
    dec1.eval()   #model in evaluation mode

    with torch.no_grad():   #evaluation mode, no autograd tracking
        #testing
        #used for train data,val_data, test_data
        if N>5120:
            n=0
            batch_size=int(N/10)  #split large dataset(train data 51200 examples(ex:after every epoch), into batch of 5120 for accuracy/loss calc),else entire data in one go(batch size=N)
        else:
            n=0
            batch_size = N
        while n<N:
                #do forward propagation
            decoder_h0,decoder_c0,encoder_hidden = encoder_forward_prop(enc1,x[n:n+batch_size],seq_len_x[n:n+batch_size])
            l,predicted_batch =decoder_forward_prop(dec1,x[n:n+batch_size],y[n:n+batch_size],seq_len_x[n:n+batch_size],
                                                        seq_len_y[n:n+batch_size],dev_script_word2idx,crit,decoder_h0,decoder_c0,
                                                        encoder_hidden,teacher_forcing_bool=False,testing=testing) 
                #update loss and predicted characters(indices)
            predicted_y[n:n+batch_size,:(max(seq_len_y[n:n+batch_size])-1) if not testing else 30] = predicted_batch
            n+=batch_size
            loss+=l

        for i in range(N):  #accuracy check(till <eow>)
            if  torch.all(y[i,1:seq_len_y[i]] == predicted_y[i,:seq_len_y[i]-1]):
                accuracy+=1
        accuracy= accuracy*100/N
        loss/=N
                
    return (loss.item(),accuracy,predicted_y) if not testing else (accuracy,predicted_y)  
#during testing/inference, only accuracy is returned(that is,if testing parameter set to True) 


# training func: trains the encoder-decoder network, with deault parameters taken from dfault_config dictionary
def training(enc,dec,dev_script_word2idx,train_data,val_data,max_length_x,max_length_val_x,
             loss_func=default_config["loss_func"],
             optimizer=default_config["optimizer"],
             max_epoch=default_config["max_epoch"],batch_size=default_config["batch_size"],learning_rate=default_config["learning_rate"],
             beta_1=default_config["beta_1"],beta_2=default_config["beta_2"],epsilon=default_config["epsilon"],
             wei_decay=default_config["wei_decay"],early_stopping=default_config["early_stopping"],
             patience=default_config["patience"],
             wandb_log=default_config["wandb_log"],teacher_forcing=default_config["teacher_forcing"]):
    
    
    enc_model_history= None
    dec_model_history=None
    incr_val_loss_count= None
    val_acc_hist= []
    prev_val_loss=float("inf")
  

    if early_stopping:          # model tracking if early_stopping is True
        enc_model_history= Queue()
        enc_model_history.put(enc.state_dict())
        dec_model_history = Queue()
        dec_model_history.put(enc.state_dict())
        incr_val_loss_count= [0]
        
    
    criterion=None
    enc_opt=None
    dec_opt=None

    if loss_func== "cross_entropy":     #set the loss criterion to cross entropy
        criterion = nn.CrossEntropyLoss(ignore_index=0,reduction="sum")
    else:
        print("loss func not supported")
    
    if optimizer=="Adam":         # set same optimers for encoder and decoder model, based on optimizer parameter
        enc_opt= optim.Adam(enc.parameters(),lr=learning_rate,betas=(beta_1,beta_2),eps=epsilon,weight_decay=wei_decay)
        dec_opt= optim.Adam(dec.parameters(),lr=learning_rate,betas=(beta_1,beta_2),eps=epsilon,weight_decay=wei_decay)

    elif optimizer== "NAdam":
        enc_opt= optim.NAdam(enc.parameters(),lr=learning_rate,betas=(beta_1,beta_2),eps=epsilon,weight_decay=wei_decay)
        dec_opt= optim.NAdam(dec.parameters(),lr=learning_rate,betas=(beta_1,beta_2),eps=epsilon,weight_decay=wei_decay)
    
    elif optimizer == "RMSprop":
        enc_opt= optim.RMSprop(enc.parameters(),lr=learning_rate,alpha=beta_1,eps=epsilon,momentum=beta_2,weight_decay=wei_decay)
        dec_opt= optim.RMSprop(dec.parameters(),lr=learning_rate,alpha=beta_1,eps=epsilon,momentum=beta_2,weight_decay=wei_decay)

    elif optimizer== "Adamax":
        enc_opt= optim.Adamax(enc.parameters(),lr=learning_rate,betas=(beta_1,beta_2),eps=epsilon,weight_decay=wei_decay)
        dec_opt= optim.Adamax(dec.parameters(),lr=learning_rate,betas=(beta_1,beta_2),eps=epsilon,weight_decay=wei_decay)

    elif optimizer == "SGD":
        enc_opt= optim.SGD(enc.parameters(),lr=learning_rate,momentum=beta_1,weight_decay=wei_decay)
        dec_opt= optim.SGD(dec.parameters(),lr=learning_rate,momentum=beta_1,weight_decay=wei_decay)

    elif optimizer == "Nesterov":
        enc_opt= optim.SGD(enc.parameters(),lr=learning_rate,momentum=beta_1,weight_decay=wei_decay,nesterov=True)
        dec_opt= optim.SGD(dec.parameters(),lr=learning_rate,momentum=beta_1,weight_decay=wei_decay,nesterov=True)
    
    else:
        print("optimizer not supported")
    
    train_loader= DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True) #batch-wise training with shuffling during each epoch
    for epoch in range(1,max_epoch+1):
        for data in train_loader:
            enc_opt.zero_grad()
            dec_opt.zero_grad()                #sets model parameter grads to zero

            batch_size=len(data)
            loss=torch.tensor(0.0).to(device=device)     
            train_x = data[:,:max_length_x].to(device=device)    
            train_y = data[:,max_length_x:-2].to(device=device)  #split training data into train x and train y
            seq_lens_train_x= data[:,-2].to(device=device)
            seq_lens_train_y= data[:,-1].to(device=device)

            decoder_c0=None
            decoder_h0=None

            #forward prop
            decoder_h0,decoder_c0,encoder_hidden = encoder_forward_prop(enc,train_x,seq_lens_train_x)
            loss,_= decoder_forward_prop(dec,train_x,train_y,seq_lens_train_x,seq_lens_train_y,dev_script_word2idx,criterion,
                                        decoder_h0,decoder_c0, encoder_hidden,epoch<= (teacher_forcing* max_epoch),testing=False)

            #backward prop

            loss.backward()

            #optimisation  step
            enc_opt.step()
            dec_opt.step()
        
        #compute train_loss,val_loss,train_acc,val_acc after each epoch
        train_loss,train_acc,_ = compute_loss_accuracy(train_data[:,:max_length_x],train_data[:,max_length_x:-2],enc,dec,
                                                       train_data[:,-2],train_data[:,-1],dev_script_word2idx,testing=False,
                                                       crit=criterion)
        val_loss,val_acc,_ = compute_loss_accuracy(val_data[:,:max_length_val_x],val_data[:,max_length_val_x:-2],enc,dec,
                                                 val_data[:,-2],val_data[:,-1],dev_script_word2idx,testing=False,
                                                 crit=criterion)
        
        val_acc_hist.append(val_acc)
        print("epoch: {}".format(epoch))
        print("=>train_acc={},train_loss={},val_acc={},val_loss={}".format(train_acc,train_loss,
                                                                                val_acc,val_loss))
        print("-"*200)
        val_acc_hist.append(val_acc)

        if wandb_log:   #if wandb_log enable, log the run
            wandb.log({"epoch":epoch,"train_accuracy":train_acc,
                        "train_loss":train_loss,
                        "val_accuracy":val_acc,
                        "val_loss": val_loss})
            
        if early_stopping:           # check for early after after every epoch
            update_history(patience,enc_model_history,dec_model_history,enc,dec)
            if do_early_stopping(patience,prev_val_loss,val_loss,incr_val_loss_count,enc_model_history,dec_model_history,enc,dec):
                    print("Early stopping event has occured!!!")
                    print("val_accuracy (before the event): ", val_acc_hist[int(-1*(patience+1))])
                    break
            prev_val_loss=val_loss
        #print/logging

        enc.train()  #sets model back to training ater accuracy,,loss computation
        dec.train()

