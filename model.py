# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --
# Main code for meta seq2seq architecture
# --

# We based the seq2seq code on the PyTorch tutorial of Sean Robertson
#   https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb 

def describe_model(net):
    if type(net) is MetaNetRNN:
        print('EncoderMetaNet specs:')
        print(' nlayers=' + str(net.nlayers))
        print(' embedding_dim=' + str(net.embedding_dim))
        print(' dropout=' + str(net.dropout_p))
        print(' bi_encoder=' + str(net.bi_encoder))
        print(' n_input_symbols=' + str(net.input_size))
        print(' n_output_symbols=' + str(net.output_size))
        print('')
    elif type(net) is AttnDecoderRNN:
        print('AttnDecoderRNN specs:')
        print(' nlayers=' + str(net.nlayers))
        print(' hidden_size=' + str(net.hidden_size))
        print(' dropout=' + str(net.dropout_p))
        print(' n_output_symbols=' + str(net.output_size))
        print('')
    elif type(net) is DecoderRNN:
        print('DecoderRNN specs:')
        print(' nlayers=' + str(net.nlayers))
        print(' hidden_size=' + str(net.hidden_size))
        print(' dropout=' + str(net.dropout_p))
        print(' n_output_symbols=' + str(net.output_size))
        print("")
    elif type(net) is EncoderRNN or type(net) is WrapperEncoderRNN:
        print('EncoderRNN specs:')
        print(' bidirectional=' + str(net.bi))
        print(' nlayers=' + str(net.nlayers))
        print(' hidden_size=' + str(net.embedding_dim))
        print(' dropout=' + str(net.dropout_p))
        print(' n_input_symbols=' + str(net.input_size))
        print('')
    else:
        print('Network type not found...')

class MetaNetRNN(nn.Module):
    # Meta Seq2Seq encoder
    #
    # Encodes query items in the context of the support set, which is stored in external memory.
    #
    #  Architecture
    #   1) RNN encoder for input symbols in query and support items (either shared or separate)
    #   2) RNN encoder for output symbols in the support items only
    #   3) Key-value memory for embedding query items with support context
    #   3) MLP to reduce the dimensionality of the context-sensitive embedding
    def __init__(self, embedding_dim, input_size, output_size, nlayers, dropout_p=0.1, bidirectional=True, tie_encoders=True):
        # 
        # Input
        #  embedding_dim : number of hidden units in RNN encoder, and size of all embeddings
        #  input_size : number of input symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers in RNN encoder
        #  dropout : dropout applied to symbol embeddings and RNNs
        #  bidirectional : use bi-directional RNN encoders? (default=True)
        #  tie_encoders : use the same encoder for the support and query items? (default=True)
        #
        super(MetaNetRNN, self).__init__()
        self.nlayers = nlayers
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.dropout_p = dropout_p
        self.bi_encoder = bidirectional
        self.attn = Attn()
        self.suppport_embedding = EncoderRNN(input_size, embedding_dim, nlayers, dropout_p, bidirectional)
        if tie_encoders:
            self.query_embedding = self.suppport_embedding
        else:    
            self.query_embedding = EncoderRNN(input_size, embedding_dim, nlayers, dropout_p, bidirectional)
        self.output_embedding = EncoderRNN(output_size, embedding_dim, nlayers, dropout_p, bidirectional)
        self.hidden = nn.Linear(embedding_dim*2,embedding_dim)
        self.tanh = nn.Tanh()

    def forward(self, sample):
        #
        # Forward pass over an episode
        #
        # Input
        #   sample: episode dict wrapper for ns support and nq query examples (see 'build_sample' function in training code)
        # 
        # Output
        #   context_last : [nq x embedding]; last step embedding for each query example
        #   embed_by_step: embedding at every step for each query [max_xq_length x nq x embedding_dim]
        #   attn_by_step : attention over support items at every step for each query [max_xq_length x nq x ns]
        #   seq_len : length of each query [nq list]
        #
        xs_padded = sample['xs_padded'] # support set input sequences; LongTensor (ns x max_xs_length)
        xs_lengths = sample['xs_lengths'] # ns list of lengths
        ys_padded = sample['ys_padded'] # support set output sequences; LongTensor (ns x max_ys_length)
        ys_lengths = sample['ys_lengths'] # ns list of lengths
        ns = xs_padded.size(0)
        xq_padded = sample['xq_padded'] # query set input sequences; LongTensor (nq x max_xq_length)
        xq_lengths = sample['xq_lengths'] # nq list of lengths
        nq = xq_padded.size(0)

        # Embed the input sequences for support and query set
        embed_xs,_ = self.suppport_embedding(xs_padded,xs_lengths) # ns x embedding_dim
        embed_xq,dict_embed_xq = self.query_embedding(xq_padded,xq_lengths) # nq x embedding_dim
        embed_xq_by_step = dict_embed_xq['embed_by_step'] # max_xq_length x nq x embedding_dim (embedding at each step)
        len_xq = dict_embed_xq['seq_len'] # len_xq is nq array with length of each sequence

        # Embed the output sequences for support set
        embed_ys,_ = self.output_embedding(ys_padded,ys_lengths) # ns x embedding_dim

        # Compute context based on key-value memory at each time step for queries
        max_xq_length = embed_xq_by_step.size(0) # for purpose of attention, this is the "batch_size"
        value_by_step, attn_by_step = self.attn(embed_xq_by_step, embed_xs.expand(max_xq_length,-1,-1), embed_ys.expand(max_xq_length, -1, -1))
            # value_by_step : max_xq_length x nq x embedding_dim
            # attn_by_step : max_xq_length x nq x ns
        concat_by_step = torch.cat((embed_xq_by_step,value_by_step),2) # max_xq_length x nq x embedding_dim*2
        context_by_step = self.tanh(self.hidden(concat_by_step)) # max_xq_length x nq x embedding_dim

        # Grab the last context for each query
        context_last = [context_by_step[len_xq[q]-1,q,:] for q in range(nq)] # list of 1D Tensors
        context_last = torch.stack(context_last, dim=0) # nq x embedding_dim
        return context_last, {'embed_by_step' : context_by_step, 'attn_by_step' : attn_by_step, 'seq_len' : len_xq}
            # context_last : nq x embedding
            # embed_by_step: embedding at every step for each query [max_xq_length x nq x embedding_dim]
            # attn_by_step : attention over support items at every step for each query [max_xq_length x nq x ns]
            # seq_len : length of each query [nq list]

class EncoderRNN(nn.Module):
    # LSTM encoder for a sequence of symbols
    #
    # The RNN hidden vector (not cell vector) at each step is captured,
    #   for transfer to an attention-based decoder.
    #
    # Does not assume that sequences are sorted by length
    def __init__(self, input_size, embedding_dim, nlayers, dropout_p, bidirectional):
        #
        # Input
        #  input_size : number of input symbols
        #  embedding_dim : number of hidden units in RNN encoder, and size of all embeddings        
        #  nlayers : number of hidden layers
        #  dropout : dropout applied to symbol embeddings and RNNs
        #  bidirectional : use a bidirectional LSTM instead and sum of the resulting embeddings
        #
        super(EncoderRNN, self).__init__()
        self.nlayers = nlayers
        self.input_size = input_size
        self.embedding_dim = embedding_dim        
        self.dropout_p = dropout_p
        self.bi = bidirectional
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers=nlayers, dropout=dropout_p, bidirectional=bidirectional)

    def forward(self, z_padded, z_lengths):
        # Input 
        #   z_padded : LongTensor (n x max_length); list of n padded input sequences
        #   z_lengths : Python list (length n) for length of each padded input sequence        
        # 
        # Output
        #   hidden is (n x embedding_size); last hidden state for each input sequence
        #   embed_by_step is (max_length x n x embedding_size); stepwise hidden states for each input sequence                 
        #   seq_len is tensor of length n; length of each input sequence
        z_embed = self.embedding(z_padded) # n x max_length x embedding_size
        z_embed = self.dropout(z_embed) # n x max_length x embedding_size

        # Sort the sequences by length in descending order
        n = len(z_lengths)
        max_length = max(z_lengths)
        z_lengths = torch.LongTensor(z_lengths)
        if z_embed.is_cuda: z_lengths = z_lengths.cuda()
        z_lengths, perm_idx = torch.sort(z_lengths, descending=True)
        z_embed = z_embed[perm_idx]

        # RNN embedding
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(z_embed, z_lengths, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_input)
             # hidden is nlayers*num_directions x n x embedding_size
             # hidden and cell are unpacked, such that they stores the last hidden state for each unpadded sequence        
        hidden_by_step, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output) # max_length x n x embedding_size*num_directions

        # If biLSTM, sum the outputs for each direction
        if self.bi:
            hidden_by_step = hidden_by_step.view(max_length, n, 2, self.embedding_dim)
            hidden_by_step = torch.sum(hidden_by_step, 2) # max_length x n x embedding_size
            hidden = hidden.view(self.nlayers, 2, n, self.embedding_dim)
            hidden = torch.sum(hidden, 1) # nlayers x n x embedding_size
        hidden = hidden[-1,:,:] # n x embedding_size (grab the last layer)

        # Reverse the sorting
        _, unperm_idx = perm_idx.sort(0)
        hidden = hidden[unperm_idx,:] # n x embedding_size
        hidden_by_step = hidden_by_step[:,unperm_idx,:] # max_length x n x embedding_size
        seq_len = z_lengths[unperm_idx].tolist()

        return hidden, {"embed_by_step" : hidden_by_step, "seq_len" : seq_len}
                # hidden is (n x embedding_size); last hidden state for each input sequence
                # embed_by_step is (max_length x n x embedding_size); stepwise hidden states for each input sequence                 
                # seq_len is tensor of length n; length of each input sequence

class WrapperEncoderRNN(EncoderRNN):
    # Wrapper for RNN encoder to behave like MetaNetRNN encoder.
    #  This isn't really doing meta-learning, since it is ignoring the support set entirely. 
    #  Instead, it allows us to train a standard sequence-to-sequence model, using the query set as the batch.
    def __init__(self, embedding_dim, input_size, output_size, nlayers, dropout_p=0.1, bidirectional=True, tie_encoders=True):
        super(WrapperEncoderRNN, self).__init__(input_size, embedding_dim, nlayers, dropout_p, bidirectional)
    def forward(self, sample):
        hidden, mydict = super(WrapperEncoderRNN, self).forward(sample['xq_padded'],sample['xq_lengths'])
        mydict['attn_by_step'] = [] # not applicable
        return hidden, mydict

class Attn(nn.Module):
    # batch attention module

    def __init__(self):
        super(Attn, self).__init__()

    def forward(self, Q, K, V):
        #
        # Input
        #  Q : Matrix of queries; batch_size x n_queries x query_dim
        #  K : Matrix of keys; batch_size x n_memory x query_dim
        #  V : Matrix of values; batch_size x n_memory x value_dim
        #
        # Output
        #  R : soft-retrieval of values; batch_size x n_queries x value_dim
        #  attn_weights : soft-retrieval of values; batch_size x n_queries x n_memory
        query_dim = torch.tensor(float(Q.size(2)))
        if Q.is_cuda: query_dim = query_dim.cuda()
        attn_weights = torch.bmm(Q,K.transpose(1,2)) # batch_size x n_queries x n_memory
        attn_weights = torch.div(attn_weights, torch.sqrt(query_dim))
        attn_weights = F.softmax(attn_weights, dim=2) # batch_size x n_queries x n_memory
        R = torch.bmm(attn_weights,V) # batch_size x n_queries x value_dim
        return R, attn_weights

class AttnDecoderRNN(nn.Module):
    #
    # One-step batch LSTM decoder with Luong et al. attention
    # 
    def __init__(self, hidden_size, output_size, nlayers, dropout_p=0.1):
        #
        # Input        
        #  hidden_size : number of hidden units in RNN, and embedding size for output symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers
        #  dropout_p : dropout applied to symbol embeddings and RNNs
        #
        super(AttnDecoderRNN, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=nlayers, dropout=dropout_p)
        self.attn = Attn()
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, last_hidden, encoder_outputs):
        #
        # Run batch decoder forward for a single time step.
        #  Each decoder step considers all of the encoder_outputs through attention.
        #  Attention retrieval is based on decoder hidden state (not cell state)
        #
        # Input
        #  input: LongTensor of length batch_size (single step indices for batch)
        #  last_hidden: previous decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #  encoder_outputs: all encoder outputs for attention, max_input_length x batch_size x embedding_size
        #
        # Output
        #   output : unnormalized output probabilities, batch_size x output_size
        #   hidden : current decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #   attn_weights : attention weights, batch_size x max_input_length 
        # 
        # Embed each input symbol
        batch_size = input.numel()
        embedding = self.embedding(input) # batch_size x hidden_size
        embedding = self.dropout(embedding)
        embedding = embedding.unsqueeze(0) # S=1 x batch_size x hidden_size

        rnn_output, hidden = self.rnn(embedding, last_hidden)
            # rnn_output is S=1 x batch_size x hidden_size
            # hidden is nlayer x batch_size x hidden_size (pair for hidden and cell)

        context, attn_weights = self.attn(rnn_output.transpose(0,1), encoder_outputs.transpose(0,1), encoder_outputs.transpose(0,1))
            # context : batch_size x 1 x hidden_size
            # attn_weights : batch_size x 1 x max_input_length
        
        # Concatenate the context vector and RNN hidden state, and map to an output
        rnn_output = rnn_output.squeeze(0) # batch_size x hidden_size
        context = context.squeeze(1) # batch_size x hidden_size
        attn_weights = attn_weights.squeeze(1) # batch_size x max_input_length        
        concat_input = torch.cat((rnn_output, context), 1) # batch_size x hidden_size*2
        concat_output = self.tanh(self.concat(concat_input)) # batch_size x hidden_size
        output = self.out(concat_output) # batch_size x output_size
        return output, hidden, attn_weights
            # output : [unnormalized probabilities] batch_size x output_size
            # hidden: pair of size [nlayer x batch_size x hidden_size] (pair for hidden and cell)
            # attn_weights: tensor of size (batch_size x max_input_length)

    def initHidden(self, encoder_message):
        # Populate the hidden variables with a message from the decoder. 
        # All layers, and both the hidden and cell vectors, are filled with the same message.
        #   message : batch_size x hidden_size tensor
        encoder_message = encoder_message.unsqueeze(0) # 1 x batch_size x hidden_size
        encoder_message = encoder_message.expand(self.nlayers,-1,-1).contiguous() # nlayers x batch_size x hidden_size tensor
        return (encoder_message, encoder_message)

class DecoderRNN(nn.Module):
    #
    # One-step simple batch LSTM decoder with no attention
    #
    def __init__(self, hidden_size, output_size, nlayers, dropout_p=0.1):
        # Input        
        #  hidden_size : number of hidden units in RNN, and embedding size for output symbols
        #  output_size : number of output symbols
        #  nlayers : number of hidden layers
        #  dropout_p : dropout applied to symbol embeddings and RNNs
        super(DecoderRNN, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=nlayers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, last_hidden):
        # Run batch decoder forward for a single time step.
        #
        # Input
        #  input: LongTensor of length batch_size 
        #  last_hidden: previous decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #
        # Output
        #   output : unnormalized output probabilities, batch_size x output_size
        #   hidden : current decoder state, which is pair of tensors [nlayer x batch_size x hidden_size] (pair for hidden and cell)
        #
        # Embed each input symbol
        batch_size = input.numel()
        embedding = self.embedding(input) # batch_size x hidden_size
        embedding = self.dropout(embedding)
        embedding = embedding.unsqueeze(0) # S=1 x batch_size x hidden_size
        rnn_output, hidden = self.rnn(embedding, last_hidden)
            # rnn_output is S=1 x batch_size x hidden_size
            # hidden is nlayer x batch_size x hidden_size (pair for hidden and cell)
        rnn_output = rnn_output.squeeze(0) # batch_size x hidden_size
        output = self.out(rnn_output) # batch_size x output_size
        return output, hidden
            # output : [unnormalized probabilities] batch_size x output_size
            # hidden: pair of size [nlayer x batch_size x hidden_size] (pair for hidden and cell)

    def initHidden(self, encoder_message):
        # Populate the hidden variables with a message from the decoder. 
        # All layers, and both the hidden and cell vectors, are filled with the same message.
        #   message : batch_size x hidden_size tensor
        encoder_message = encoder_message.unsqueeze(0) # 1 x batch_size x hidden_size
        encoder_message = encoder_message.expand(self.nlayers,-1,-1).contiguous() # nlayers x batch_size x hidden_size tensor
        return (encoder_message, encoder_message)