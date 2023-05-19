# cs6910_assignment_3

* data_loader_vocab_builder.py : 

i)The python script loads the aksharantar tamil dataset,builds the character dictionary(python dictionary)for both latin and devanagari script) and converts the characters into torch tensor with character indices(index-> character mapping is done using the built dictionaries).

ii) Returns the built character2idx and idx2character dictionaries and the torch tensors: train_data(training data),val_data(validation data) and test_data(testing data)

* encoder_decoder.py: 

i) Contains the encoder,decoder and attention decoder class definitions whose object are the encoder and decoder models for the seq2seq network.

ii) Encoder Network:
  
  * Case1 (bidirectional-False): Input->Embedding->Dropout->cell(rnn/gru/lstm/)->hidden states(returned during forward())
  * Case2 (bidirectional-True): Input->Embedding->Dropout->cell(rnn/gru/lstm/)->2xhidden states->linear layer-> hidden states (returned during forward())
  * h0,c0 of the cell-  are taken as trainable parameters
 
 iii) Decoder Network:
   * The number of hidden layers of encoder network(cell) is same as that of decoder network inorder to prevent bottleneck architecture during passage of hidden states from encoder to decoder.
   * (Input character/prev predicted character)->Embedding->Dropout->cell(rnn/gru/lstm)->linear layer->Predicted character. 
   * No softmax layer after linear layer as nn.Crossentropyloss() is used during training and testing which directly takes in logits.
   * The hidden layer size of both the encoder and decoder cells are the same
   * A uniform dropout probability is used for all the dropout layers in both encoder and decoder network.
 
 iv) Attention Decoder Network:
     * (Input character/prev predicted character) + encoder hidden states * attn weights)->Linear Layer-> combined input->Embedding->Dropout->cell(rnn/gru/lstm)->linear layer->Predicted character. 
     * Attention weights are obtained using Watt(linear layer),Uatt(linear layer) and Vatt(trainable vector for dot product)=> $Vatt.(Watt(h_j) +Uatt(s_t-1) $
 
* main.py:

 i) Contains the definitions of training() function(network training:forward,backward with early stopping) and compute_loss_accuracy() function (for calculating loss,accuracy for training data,validation data and test data).
 
 * hyperarameter_tuning.py:

  i) wandb sweep experiments: hyperparameter_tuning1,hyperparameter_tuning2 and hyperparameter_tuning3
  ii) The hyperparameter sweeps were run in kaggle with aksharantar dataset added as data source,encoder_decoder.py,data_loader_vocab_builder.py and main.py added as utility scripts for the kaggle notebook.

 * train.py: (Tamil dataset)

  i) Python script to train encoder-decoder seq2seq model with training data and test the model on the test dataset from the terminal/command prompt.
  ii) Commandline arguments are parsed using "argparse" module.
  iii) Type: python train.py --help in the terminal to get info about the commandline arguments. The default values of the arguments are set to the best configuration for the optimizer and model obtained from sweep configuration.
  

  
     
 
   
 
 
 
 
