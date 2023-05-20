import torch
from encoder_decoder import encoder,decoder,attention_decoder
from data_loader_vocab_builder import *
import wandb
from main_def import training

wandb.login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #check for cuda
torch.manual_seed(42) #global random seed

(latin_script,latin_script_idx2word,tam_script_word2idx,tam_script_idx2word),(
    train_data,max_length_x,max_length_y),(val_data,max_length_val_x,max_length_val_y),(
    test_data,max_length_test_x,max_length_test_y)=load_data()   #load data and script dictionaries

train_data = train_data.to(device=device)
val_data = val_data.to(device=device)


def hyperparametric_tuning():
    #default model and training config
    default_config={"cell_type":"rnn","hidden_size":256,"enc_embedding_size":256,"enc_dict_size": len(latin_script_idx2word),
                       "num_hl":2,"bidirectional":False,"dec_embedding_size":256,
                       "dec_dict_size": len(tam_script_word2idx),"loss_func":"cross_entropy","optimizer":"Adam",
                        "max_epoch":1,"batch_size":256,"learning_rate":0.002,
                        "beta_1":0.99,"beta_2":0.999,"epsilon":10**-8,"wei_decay":0.005,
                        "early_stopping":True,"patience":4,"wandb_log":True,"teacher_forcing":0.0,"attn_mech":False,"dropout":0.2}
    
    run=wandb.init(project="assignment3_cs6910",config=default_config)

    config=wandb.config
    #setting up wandb run with name as given
    sweep_name="ep_{}_bs_{}_hlnum_{}_hlsize_{}_cell_{}_bidir_{}_eemsize_{}_demsize_{}_tf_{},drop_{}_lr_{}_attn_{}".format(config.max_epoch,config.batch_size,
                                                                                                 config.num_hl,
                                                                                                 config.hidden_size,config.cell_type,
                                                                                                 config.bidirectional,config.enc_embedding_size,
                                                                                                 config.dec_embedding_size,config.teacher_forcing,
                                                                                                 config.dropout,config.learning_rate,config.attn_mech
                                                                                                 )
    
    run.name=sweep_name

    #initialise encoder network
    enc = encoder(hidden_size=config.hidden_size,num_of_hidden_layers=config.num_hl,dict_size=config.enc_dict_size,
               embedding_size= config.enc_embedding_size,cell_type= config.cell_type,
               bidirectional= config.bidirectional,dropout= config.dropout)
    enc.to(device=device)
    #initialise decoder with or without attn mech
    if not config.attn_mech:
        dec = decoder(config.hidden_size,
        num_of_hidden_layers= config.num_hl,dict_size= config.dec_dict_size,
        embedding_size= config.dec_embedding_size,
        cell_type= config.cell_type,dropout=config.dropout)
    
    else:
        dec = attention_decoder(hidden_size= config.hidden_size,
                            num_of_hidden_layers=config.num_hl,dict_size=config.dec_dict_size,
                            embedding_size= config.dec_embedding_size,
                            cell_type=config.cell_type,
                            encoder_hidden_size=config.hidden_size,
                            dropout= config.dropout)
    dec.to(device=device)
    
    #call training from main script for training the encoder-decoder network with parameters from config
    training(enc,dec,tam_script_word2idx,train_data,val_data,max_length_x,max_length_val_x,loss_func=config.loss_func,
             optimizer= config.optimizer,
             max_epoch= config.max_epoch,batch_size= config.batch_size,learning_rate= config.learning_rate,
             beta_1= config.beta_1,beta_2= config.beta_2,epsilon= config.epsilon,
             wei_decay= config.wei_decay,early_stopping=config.early_stopping,
             patience=config.patience,
             teacher_forcing= config.teacher_forcing,wandb_log= config.wandb_log)
    #save trained model state_dixt for loading
    torch.save( enc.state_dict(),"/kaggle/working/enc_{}.pth".format(sweep_name))
    torch.save( dec.state_dict(),"/kaggle/working/dec_{}.pth".format(sweep_name))

#hyperparameter tuning 1 with large space and attn_mech=False

sweep_configuration_1={
    "project": "assignment3_cs6910",
    "method" : "bayes",
    "name" : "hyperparameter_tuning_1",
    "metric": {
        "goal": "maximize",
        "name": "val_accuracy"
    },
    "run_cap":100,
    "parameters":{
        "cell_type": {"values": [ "rnn","gru","lstm"] },

        "max_epoch": {"values": [10,15,20,30]},

        "batch_size": {"values": [32,64,128]},

        "num_hl": {"values": [1,2,3,4]},

        "hidden_size": {"values": [32,64,128,256]},

        "bidirectional" : { "values": [True,False]},

        "enc_embedding_size": { "values": [64,128,256]},

        "dec_embedding_size": { "values" : [64,128,256]},

        "teacher_forcing" : { "values" : [0,0.2,0.4,0.5,0.6,0.7]},

        "dropout" : { "values" : [ 0,0.15,0.3,0.45]},

        "learning_rate" : { "values": [0.002,0.0002]}
    }
}
sweep_id= wandb.sweep(sweep_configuration_1,project="assignment3_cs6910")
wandb.agent(sweep_id,function=hyperparametric_tuning)

#hyperparameter tuning 2 with reduced space(based on sweep 1) and attn_mech=False
sweep_configuration_2={
    "project": "assignment3_cs6910",
    "method" : "random",
    "name" : "hyperparameter_tuning_2",
    "metric": {
        "goal": "maximize",
        "name": "val_accuracy"
    },
    "run_cap":90,
    "parameters":{
        "cell_type": {"values": [ "lstm"] },

        "max_epoch": {"values": [20,30]},
    
        "batch_size": {"values": [32,64]},

        "num_hl": {"values": [3,4]},
    
        "hidden_size": {"values": [128,256]},

        "bidirectional" : { "values": [False,True]},

        "enc_embedding_size": { "values": [128,256]},

        "dec_embedding_size": { "values" : [128,256]},

        "teacher_forcing" : { "values" : [0.5,0.6,0.7]},

        "dropout" : { "values" : [ 0.3,0.45]},

        "learning_rate" : { "values": [0.002,0.0002]}
    }
}
sweep_id= wandb.sweep(sweep_configuration_2,project="assignment3_cs6910")
wandb.agent(sweep_id,function=hyperparametric_tuning)

#hyperparameter tuning 3 with space(same as sweep 2) and attn_mech=True
sweep_configuration_3={
    "project": "assignment3_cs6910",
    "method" : "random",
    "name" : "hyperparameter_tuning_3",
    "metric": {
        "goal": "maximize",
        "name": "val_accuracy"
    },
    "run_cap":100,
    "parameters":{
        "cell_type": {"values": [ "lstm"] },
        "max_epoch": {"values": [20,30]},

        "batch_size": {"values": [32,64]},
        "num_hl": {"values": [3,4]},

        "hidden_size": {"values": [128,256]},
        "bidirectional" : { "values": [True,False]},

        "attn_mech": {"values": [True]},
        "enc_embedding_size": { "values": [128,256]},
        "dec_embedding_size": { "values" : [128,256]},

        "teacher_forcing" : { "values" : [0.5,0.6,0.7]},
        "dropout" : { "values" : [ 0.45,0.3]},

        "learning_rate" : { "values": [0.002,0.0002]}
    }
}
sweep_id= wandb.sweep(sweep_configuration_3,project="assignment3_cs6910")
wandb.agent(sweep_id,function=hyperparametric_tuning)
