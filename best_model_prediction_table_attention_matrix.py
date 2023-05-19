import torch
import numpy as np
from encoder_decoder import encoder,decoder,attention_decoder
from data_loader_vocab_builder import *
from train import compute_loss_accuracy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

#loading test data and vocab dictionary
(latin_script_word2idx,latin_script_idx2word,dev_script_word2idx,dev_script_idx2word),(
    _,_,_),(_,_,_),(
    test_data,max_length_test_x,max_length_test_y)=load_data()

test_data=test_data.to(device=device)

#loading the best models obtained from the sweep (without attention mech)-sweeps run in kaggle and the models saved in kaggle
best_enc_no_attn= encoder(hidden_size=256,num_of_hidden_layers=4,dict_size=len(latin_script_idx2word),bidirectional=False,dropout=0.3,
                   embedding_size=256,cell_type="lstm")
best_enc_no_attn.load_state_dict(torch.load("/kaggle/input/best-no-attn/enc_ep_30_bs_64_hlnum_4_hlsize_256_cell_lstm_bidir_False_eemsize_256_demsize_128_tf_0.6drop_0.3_lr_0.002_attn_False.pth"))
best_dec_no_attn = decoder(hidden_size=256,num_of_hidden_layers=4,cell_type="lstm",dict_size=len(dev_script_idx2word),dropout=0.3,
                                   embedding_size=128)
best_dec_no_attn.load_state_dict(torch.load("/kaggle/input/best-no-attn/dec_ep_30_bs_64_hlnum_4_hlsize_256_cell_lstm_bidir_False_eemsize_256_demsize_128_tf_0.6drop_0.3_lr_0.002_attn_False.pth"))

#loading the best models obtained from the sweep (with attention mech)-sweeps run in kaggle and the models saved in kaggle

best_enc_attn= encoder(hidden_size=256,num_of_hidden_layers=3,dict_size=len(latin_script_idx2word),bidirectional=True,dropout=0.3,
                   embedding_size=256,cell_type="lstm")
best_enc_attn.load_state_dict(torch.load("/kaggle/input/best-attn/enc_ep_30_bs_32_hlnum_3_hlsize_256_cell_lstm_bidir_True_eemsize_256_demsize_128_tf_0.6drop_0.3_lr_0.0002_attn_True.pth"))
best_dec_attn = attention_decoder(hidden_size=256,num_of_hidden_layers=3,cell_type="lstm",dict_size=len(dev_script_idx2word),dropout=0.3,
                                   embedding_size=128,encoder_hidden_size=256)
best_dec_attn.load_state_dict(torch.load("/kaggle/input/best-attn/dec_ep_30_bs_32_hlnum_3_hlsize_256_cell_lstm_bidir_True_eemsize_256_demsize_128_tf_0.6drop_0.3_lr_0.0002_attn_True.pth"))


#loading to cuda if available
best_enc_no_attn.to(device=device)
best_dec_no_attn.to(device=device)
best_enc_attn.to(device=device)
best_dec_attn.to(device=device)

#setting the models in eval mode
best_enc_no_attn.eval()
best_dec_no_attn.eval()
best_enc_attn.eval()
best_dec_attn.eval()


def predictions(enc,dec,test_data,max_length_test_x,lat_dict_idx2word,lat_dict_word2idx,dev_dict_idx2word,
                dev_dict_word2idx,ignore_accuracy = False,
                beam_search=0,wandb_log_table= False):
    
    N=len(test_data)
    predicted_words_list=[]   #predicted(dev)
    actual_words_list=[]    #ground truth (dev)
    input_words_list=[]    #latin

    if not ignore_accuracy:  #compute accuracy and the predicted word indices
        acc,predicted_words = compute_loss_accuracy(test_data[:,:max_length_test_x],test_data[:,max_length_test_x:-2],enc,dec,
                                                         test_data[:,-2],test_data[:,-1],dev_dict_word2idx,testing=True,beam_search=beam_search)
    else:
        _,predicted_words = compute_loss_accuracy(test_data[:,:max_length_test_x],test_data[:,max_length_test_x:-2],enc,dec,
                                                    test_data[:,-2],test_data[:,-1],dev_dict_word2idx,testing=True,beam_search=beam_search)
    
    actual_words = test_data[:,max_length_test_x:-2]
    input_words = test_data[:,:max_length_test_x]

    for i in range(N):      #convert predicted indices(y) to  dev characters
        a=""
        for idx in predicted_words[i]:
            if idx.item() == dev_dict_word2idx["<eow>"]:
                break
            a+=dev_dict_idx2word[idx.item()]
        predicted_words_list.append(a)

    for i in range(N):      #convert ground truth indices(y) to  dev characters
        a=""
        for idx in actual_words[i]:
            if idx.item()== dev_dict_word2idx["<sow>"]:
                continue
            elif idx.item() == dev_dict_word2idx["<eow>"]:
                break
            a+=dev_dict_idx2word[idx.item()]
        actual_words_list.append(a)
    
    for i in range(N):     #convert input indices(x) to  dev characters
        a=""
        for idx in input_words[i]:
            if idx.item() == lat_dict_word2idx["<eow>"]:
                break
            a+=lat_dict_idx2word[idx.item()]
        input_words_list.append(a)
        
    heading=np.array(["Input(Latin script)","Ground Truth(Dev script)","Prediction(Dev script)"])
    result = np.vstack((np.array(input_words_list),np.array(actual_words_list),np.array(predicted_words_list))).T
    result=np.vstack((heading,result))
    # save the predictions as csv file
    if not dec.attn_mech:
        np.savetxt("/kaggle/working/predictions_vanilla.csv",result,delimiter=",",fmt="%s")
    else:
        np.savetxt("/kaggle/working/predictions_attention.csv",result,delimiter=",",fmt="%s")
    
    #log the predictions(selected words) as a plotly table to logged in wandb account (2 tables for no attention and 1 for with attention)
    if wandb_log_table:
        if not dec.attn_mech:
            title="Prediction Table: Without attention mechanism(Green:Right Predictions,Red:Wrong Prediction)"
            selected_words=[7,15,26,31,34,80,220,223,241,266,269,302,329,391,433,454,505,546,566,591] 
            selected_words1=[13,14,36,40,51,57,78,80,91,212,97,100]
        else:
            title="Prediction Table: With attention mechanism(Green:Right Predictions,Red:Wrong Prediction)"
            selected_words=[13,14,36,40,51,57,78,80,91,212,97,100]
        
        wandb.init(project= "assignment3_cs6910",name="prediction table {} attention".format("with" if dec.attn_mech else "without"))
        #logging 2  tables for no attention and 1 with attention
        for k in range(2 if not dec.attn_mech else 1):
            correct_prediction = np.array(actual_words_list)[selected_words if k==0 else selected_words1]==np.array(predicted_words_list)[selected_words if k==0 else selected_words1]
            fig = go.Figure(data=[go.Table(
                header=dict(values= ["S.No","Input(Latin script)","Ground Truth(Dev script)","Prediction(Dev script)"],
                            align='left'),
                cells=dict(values=[list(range(1,len(selected_words if k==0 else selected_words1)+1)), 
                                   list(np.array(input_words_list)[selected_words if k==0 else selected_words1]),list(np.array(actual_words_list)[selected_words if k==0 else selected_words1]),
                                   list(np.array(predicted_words_list)[selected_words if k==0 else selected_words1])],
                           fill_color=  [[ "lightgreen" if correct else "lightsalmon" for correct in correct_prediction]*4],
                           align='left',height=25))
                                 ])
            fig.update_layout(width=500, height=800)
            wandb.log({title: fig})

        wandb.finish()
    
    return predicted_words_list if ignore_accuracy else (predicted_words_list,acc)


def plot_heat_map(fig,data,x,y,row,col):
    duplicate_prevention={0:"*",1:"@",2:"#",3:"$",4:"&"} #to prevent clubbing of heatmap columns or rows base on duplicate labels
    count=dict()
    count1=dict()
    for i in range(len(x)):
        if x[i] not in count:
            count[x[i]]=1
        else:
            count[x[i]]+=1
            x[i]=duplicate_prevention[count[x[i]]]+x[i]
    for i in range(len(y)):
        if y[i] not in count1:
            count1[y[i]]=1
        else:
            count1[y[i]]+=1
            y[i]=duplicate_prevention[count1[y[i]]]+y[i]

    data=torch.round(data,decimals=2) #rounding of the weights to 2 decimal places
    fig.add_trace(go.Heatmap(z=data, x=x, y=y, coloraxis = "coloraxis"), row=row, col=col) #add a subplot for the attention matrix for the given datapoint
    fig.update_xaxes(title=dict(text="Input(Latin Script)", font=dict(size=14)), row=row, col=col, side='top')
    fig.update_yaxes(title=dict(text="Predicted(Devanagiri Script)", font=dict(size=14)), row=row, col=col, side='left',autorange="reversed")
    
    return fig


def attention_heatmap(enc,dec,test_data,max_length_test_x,dev_script_word2idx,dev_script_idx2word,
                      latin_script_idx2word,sample_size=9):
    N=len(test_data)
    permute = torch.randperm(N)
    #sample a random set of 9 words(datapoints) from the test data 
    test_data_sampled =  (test_data[permute])[:sample_size]
    test_data_x = test_data_sampled[:,: max_length_test_x]
    test_data_xseq_lens = test_data_sampled[:,-2]

    run=wandb.init(project="assignment3_cs6910")
    run.name="{}".format("Attention Heatmap")
    fig =make_subplots(rows=3,cols=3)    #create a 3x3 plotly subplot
    with torch.no_grad():
        for i in range(sample_size):  #do a forward pass for each of the 9 datapoint
            attention_weight_matrix=torch.zeros(test_data_xseq_lens[i],30)
            prediction_list = []
            if enc.cell_type=="lstm": #encoder forward pass
                (decoder_h0,decoder_c0),encoder_hidden = enc(test_data_x[i].unsqueeze(0),test_data_xseq_lens[i].unsqueeze(0))
            else:
                decoder_h0,encoder_hidden = enc(test_data_x[i].unsqueeze(0), test_data_xseq_lens[i].unsqueeze(0))

            attn_mask=torch.ones(test_data_xseq_lens[i])
            attn_mask=attn_mask.to(device=device)

            input= torch.tensor([[dev_script_word2idx["<sow>"]]])
            input=input.to(device=device)
            for j in range(30):   #decoder forward pass upto a maximum seq len of 30
                probabilities,a = dec(input,decoder_h0,decoder_c0,encoder_hidden,attn_mask) #decoder forward pass
                _,predicted = torch.max(probabilities,dim=2)
                attention_weight_matrix[:,j] = dec.attn_weights[:test_data_xseq_lens[i]] #update the attention weight matrix(attn weights are an attriv=bute of the attention_decoder class )
                prediction_list.append(dev_script_idx2word[predicted[0,0].item()])

                if predicted.item() == dev_script_word2idx["<eow>"]: #stop prediction once <eow> is predicted
                    break
                input = predicted
                if dec.cell_type=="lstm":
                    decoder_h0 = a[0]
                    decoder_c0=a[1]
                else:
                    decoder_h0 =a
        
            attention_weight_matrix = attention_weight_matrix[:,:j+1]  
            x_list = [latin_script_idx2word[idx.item()] for idx in test_data_x[i,:test_data_xseq_lens[i]]] #input sequence(x)
            
            fig=plot_heat_map(fig,attention_weight_matrix,prediction_list,x_list,row=i//3 +1,col=i%3 +1) #plotting the heatmap 
        fig.update_layout(coloraxis = {'colorscale':'viridis'}, height=1000, width=1000) 
        wandb.log({"Attention Heatmap Grid": fig})
        wandb.finish()
            

with torch.no_grad(): #calling predictions() for csv file, test acc calc and logging the prediction table 
    _,test_acc1=predictions(best_enc_no_attn,best_dec_no_attn,test_data,max_length_test_x,latin_script_idx2word,latin_script_word2idx,
                           dev_script_idx2word,dev_script_word2idx,ignore_accuracy=False,beam_search=0,wandb_log_table=True)
    _,test_acc2=predictions(best_enc_attn,best_dec_attn,test_data,max_length_test_x,latin_script_idx2word,latin_script_word2idx,
                           dev_script_idx2word,dev_script_word2idx,ignore_accuracy=False,beam_search=0,wandb_log_table=True)
print("-"*200)
print("=>Test Accuracy(no_atten_mechanism): {}".format(test_acc1))
print("-"*200)
print("=>Test Accuracy(atten_mechanism): {}".format(test_acc2))
print("-"*200)

#invoking the attention_heatmap() to log the 3x3 attention heatmap grid
attention_heatmap(best_enc_attn,best_dec_attn,test_data,max_length_test_x,dev_script_word2idx,dev_script_idx2word,latin_script_idx2word)