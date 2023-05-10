import argparse
parser=argparse.ArgumentParser(prog="Training & Testing NN model",description="""Training and Testing the encoder-decoder network on the 
given aksharantar dataset.All parse arguments have default values and are set to the best configuration obtained from wandb sweep experiments""")
parser.add_argument('-wp','--wandb_project',default="cs6910_assignment3",
                    help="Project name used to track experiments.Default set to 'cs6910_assignment3'.")
parser.add_argument('-we','--wandb_entity',default="ep19b005",
                    help="Wandb entity used to track experiments in the W&B dashboard.Default set to 'ep19b005'.")
parser.add_argument('-e','--max_epoch',default=15,type=int,help="Number of epochs to train the network.Supported type 'int'.Default set to 15")
parser.add_argument('-b','--batch_size',default=64,type=int,
                    help="Batch size used to train the network.Supported type 'int'.Default set to 64")
parser.add_argument('-l','--loss',default="cross_entropy",choices=["cross_entropy"],
                    help="""Loss function used to train the neural network.Only supported choice is 'cross_entropy'.Default set to 'cross_entropy'""")
parser.add_argument('-o','--optimizer',default="Adam",choices=["SGD","Nesterov","Adamax","RMSprop","NAdam","Adam"],help="""Optimizer used for training.Supported choices are 'SGD','Nesterov',
'Adamax','RMSprop','Adam','NAdam'.Default set to 'Adam'""")
parser.add_argument('-lr','--learning_rate',default=0.002,type=float,help="Learning rate used to optimize model parameters.Supported type 'float'.Default set to 0.00025")
parser.add_argument('-beta1','--beta1',default=0.99,type=float,help="""Beta1 used by Adam,Nadam and Adamax optimizer for beta1. 
Beta1 used by SGD and Nesterov as momentum parameter.Beta1 used by RMSprop as alpha parameter.Supported type 'float'.Default set to 0.99""")
parser.add_argument('-beta2','--beta2',default=0.999,type=float,help="Beta2 used by Adam,Nadam and Adamax optimizer.Supported type 'float'.Default set to 0.999")
parser.add_argument('-eps','--epsilon',default=10**-8,type=float,help="Epsilon used by optimizers.Supported type 'float'.Default set to 10**-8")
parser.add_argument('-w_d','--weight_decay',default=0.001,type=float,help="L2 regularisation parameter or decay parameter used during training.Supported type 'float'.Default set to 0.001")
parser.add_argument('-hl','--num_of_hidden_layers',default=5,type=int,help="""Number of hidden layers in the encoder-decoder network.Both encoder and decoder models take 
the same number of hidden layers to prevent 'bottleneck' during passing of hidden states..Supported type 'int'.Default set to 5""")
parser.add_argument('-sz','--hidden_size',default=64,type=int,help="Size of the hidden layer for both encoder & decoder networks.Supported type 'int'.Default set to 64.")
parser.add_argument('-cell','--cell_type',default="rnn",choices=["rnn","gru","lstm"],help=""" The type of encoder decoder architecture 
or hidden cell type.For relu activation.Default set to 'rnn'.""")
parser.add_argument('-enc_em','--enc_embed_size',type=int,default=256,help="""The embedding size used by encoder network.Supported type 'int'.Default set to 256.""")
parser.add_argument('-dec_em','--dec_embed_size',type=int,default=256,help="""The embedding size used by decoder network.Supported type 'int'.Default set to 256.""")
parser.add_argument('-drop','--dropout',type=float,default=0.2,help="""The probability of switch-off in the dropout layers used in encoder and decoder networks.
Supported type 'float' from 0 to 1.Default set to 0.3""")
parser.add_argument('-bi','--bidirectional',dest="bidirectional",action="store_false",help="""Boolean flag (action='store_false') with a default True value. Implements a bidirectional encoder network by default.
When flagged/called(becomes false),encoder network/model is no longer has bi-directional information/capability""")
parser.add_argument('-es','--early_stopping',dest="early_stopping",action="store_true",help="""Boolean flag (action='store_true') with a default False value.Do early stopping during training by tracking validation 
losses when set to True""")
parser.add_argument('-pat','--patience',default=4,type=int,help="""patience to be used while tracking validation loss for early stopping.Put to use only if 
'early_stopping argument' is set to True.Supported type 'int'.Default set to 4. """)
parser.add_argument('-no_log','--no_wandb_log',dest="wandb_log",action='store_false',help=""" Boolean flag (action='store_false') with a default True value.It allows logging of training loss,training accuracy,
validation accuracy and validation loss (to a already initialized wandb run) during training of Neural Network. When called/flagged(becomes False), only printing of these values occur.
.Default set to True""" )
parser.add_argument('-tf','--teacher_forcing',default=0.6,type=float,help=""" Sets the percentage of the total no. of epochs for which teacher forcing
will be enabled during training.Supported type 'float':neg values,no forcing & values greater than or equal to 1,forcing during all the epochs(entire training).
Default set to 0.6""")
parser.add_argument('-beam','--beam_search',default=0,type=int,help=""" When set to 0,does not do beam search during testing/inference on test data.When
set to non-zero positive integer values, does beam search during testing/inference with the provided integer value as beam width.
Supported type 'int':Non-negative integers.Default set to 0""")
args=parser.parse_args('-att','--attn_mech',dest="attn_mech",action="store_false",help="""Boolean flag (action='store_false') with a default True value.
Implements a decoder network with attn_mechanism by default.When called/flagged(becomes false),decoder without attention mechanism is implemented.Default set to True""")

from encoder_decoder import encoder,decoder,attention_decoder
from data_loader_vocab_builder import *
from main import training,compute_loss_accuracy

print("loading Tamil dataset")
(latin_script,latin_script_idx2word,dev_script_word2idx,dev_script_idx2word),(
    train_data,max_length_x,max_length_y),(val_data,max_length_val_x,max_length_val_y),(
    test_data,max_length_test_x,max_length_test_y)=load_data()
print("dataset_loaded")



if args.wandb_log:
    import wandb
    wandb.login()
    configuration={
        "max_epoch": args.max_epoch,
        "batch_size": args.batch_size,
        "num_of_hidden_layers": args.num_of_hidden_layers,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "beta1":args.beta1,
        "beta2":args.beta2,
        "epsilon":args.epsilon,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "loss": args.loss,
        "early_stopping": args.early_stopping,
        "patience": args.patience,
        "cell_type":args.cell_type,
        "enc_embedding_size": args.enc_embed_size,
        "dec_embedding_size": args.dec_embed_size,
        "attn_mech": args.attn_mech,
        "early_stopping": args.early_stopping,
        "patience": args.patience,
        "dropout":args.dropout,
        "teacher_forcing":args.teacher_forcing,
        "bidirectional":args.bidirectional,
        "enc_dict_size": len(latin_script),
        "dec_dict_size": len(dev_script_word2idx),
        }
    run=wandb.init(project=args.wandb_project,entity=args.wandb_entity,config=configuration)
    config=wandb.config
    name="ep_{}_bs_{}_hlnum_{}_hlsize_{}_cell_{}_bidir_{}_eemsize_{}_demsize_{}_tf_{},drop_{}_lr_{}_attn_{}_es_{}_pat_{}_wd_{}".format(config.max_epoch,config.batch_size,
                                                                                                 config.num_of_hidden_layers,
                                                                                                 config.hidden_size,config.cell_type,
                                                                                                 config.bidirectional,config.enc_embedding_size,
                                                                                                 config.dec_embedding_size,config.teacher_forcing,
                                                                                                 config.dropout,config.learning_rate,config.attn_mech,
                                                                                                 config.early_stopping,config.patience,config.weight_decay
                                                                                                 )
    run.name=name


enc = encoder(hidden_size=args.hidden_size,num_of_hidden_layers=args.num_of_hidden_layers,dict_size=len(latin_script),
              embedding_size= args.enc_embed_size,cell_type= args.cell_type,
              bidirectional= args.bidirectional,dropout= args.dropout)
enc.to(device=device)
    
if not args.attn_mech:
    dec = decoder(args.hidden_size,
    num_of_hidden_layers= args.num_of_hidden_layers,dict_size= len(dev_script_word2idx),
    embedding_size= args.dec_embedd_size,
    cell_type= args.cell_type,dropout=args.dropout)
    
else:
    dec = attention_decoder(hidden_size= args.hidden_size,
                            num_of_hidden_layers=args.num_of_hidden_layers,dict_size=len(dev_script_idx2word),
                            embedding_size= args.dec_embedd_size,
                            cell_type=args.cell_type,
                            encoder_hidden_size=args.hidden_size,
                            dropout= args.dropout)
dec.to(device=device)


training(enc=enc,dec=dec,dev_script_word2idx=dev_script_word2idx,train_data=train_data,val_data=val_data,max_length_x=max_length_x,
         max_length_val_x=max_length_val_x,loss_func=args.loss,optimizer=args.optimizer,max_epoch=args.max_epoch,batch_size=args.batch_size,
         learning_rate=args.learning_rate,beta_1=args.beta1,beta_2=args.beta_2,epsilon=args.epsilon,wei_decay=args.weight_decay,
         early_stopping=args.early_stopping,patience=args.patience,wandb_log=args.wandb_log,teacher_forcing=args.teacher_forcing)

if args.wandb_log:
    wandb.finish()
else:
    pass

acc,_=compute_loss_accuracy(x=test_data[:,:max_length_test_x],y=test_data[:,max_length_test_x:-2],enc1=enc,dec1=dec,seq_len_x=test_data[:,-2],
                      seq_len_y=test_data[:,-1],dev_script_word2idx=dev_script_word2idx,testing=True,beam_search=args.beam_search)

print("Tamil Dataset")
print("=>{} Test Accuracy:",acc)
print("-"*200)
