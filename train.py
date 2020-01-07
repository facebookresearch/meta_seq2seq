# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import argparse
import random
import os
from copy import deepcopy, copy
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import MetaNetRNN, AttnDecoderRNN, DecoderRNN, describe_model, WrapperEncoderRNN
from masked_cross_entropy import *
import generate_episode as ge

# --
# Main routine for training meta seq2seq models
# --

# We based the seq2seq code on the PyTorch tutorial of Sean Robertson
#   https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb 

USE_CUDA = torch.cuda.is_available()

# Special symbols
SOS_token = "SOS" # start of sentence
EOS_token = "EOS" # end of sentence
PAD_token = SOS_token # padding symbol

class Lang:
    # Class for converting strings/words to numerical indices, and vice versa.
    #  Should use separate class for input language (English) and output language (actions)
    #
    def __init__(self, symbols):
        # symbols : list of all possible symbols
        n = len(symbols)
        self.symbols = symbols
        self.index2symbol = {n: SOS_token, n+1: EOS_token}
        self.symbol2index = {SOS_token : n, EOS_token : n+1}
        for idx,s in enumerate(symbols):
            self.index2symbol[idx] = s
            self.symbol2index[s] = idx
        self.n_symbols = len(self.index2symbol)

    def variableFromSymbols(self, mylist, add_eos=True):
        # Convert a list of symbols to a tensor of indices (adding a EOS token at end)
        # 
        # Input
        #  mylist : list of m symbols
        #  add_eos : true/false, if true add the EOS symbol at end
        #
        # Output
        #  output : [m or m+1 LongTensor] indices of each symbol (plus EOS if appropriate)
        mylist = copy(mylist)
        if add_eos:
            mylist.append(EOS_token)
        indices = [self.symbol2index[s] for s in mylist]
        output = torch.LongTensor(indices)
        if USE_CUDA:
            output = output.cuda()
        return output

    def symbolsFromVector(self, v):
        # Convert indices to symbols, breaking where we get a EOS token
        # 
        # Input
        #  v : list of m indices
        #   
        # Output
        #  mylist : list of m or m-1 symbols (excluding EOS)
        mylist = []
        for x in v:
            s = self.index2symbol[x]
            if s == EOS_token:
                break
            mylist.append(s)
        return mylist

# Robertson's asMinutes and timeSince helper functions to print time elapsed and estimated time
# remaining given the current time and progress

def asMinutes(s): 
    # convert seconds to minutes
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    # prints time elapsed and estimated time remaining
    #
    # Input 
    #  since : previous time
    #  percent : amount of training complete
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def make_hashable(G):
    # Separate and sort stings, to make unique string identifier for an episode
    #
    # Input
    #   G : string of elements separate by \n, specifying the structure of an episode 
    G_str = str(G).split('\n')
    G_str.sort()
    out = '\n'.join(G_str)
    return out.strip()

def tabu_update(tabu_list,identifier):
    # Add all elements of "identifier" to the 'tabu_list', and return updated list
    if isinstance(identifier,(list,set)):
        tabu_list = tabu_list.union(identifier)
    elif isinstance(identifier,str):
        tabu_list.add(identifier)
    else:
        assert False
    return tabu_list

def get_unique_words(sentences):
    # Get a list of all the unique words in a list of sentences
    # 
    # Input
    #  sentences: list of sentence strings
    # Output
    #   words : list of all unique words in sentences
    words = []
    for s in sentences:
        for w in s.split(' '): # words
            if w not in words:
                words.append(w)
    return words

def pad_seq(seq, max_length):
    # Pad sequence with the PAD_token symbol to achieve max_length
    #
    # Input
    #  seq : list of symbols
    #
    # Output
    #  seq : padded list now extended to length max_length
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def build_padded_var(list_seq, lang):
    # Transform python list to a padded torch tensor
    # 
    # Input
    #  list_seq : list of n sequences (each sequence is a python list of symbols)
    #  lang : language object for translation into indices
    #
    # Output
    #  z_padded : LongTensor (n x max_length)
    #  z_lengths : python list of sequence lengths (list of scalars)
    n = len(list_seq)
    if n==0: return [],[]
    z_eos = [z+[EOS_token] for z in list_seq]
    z_lengths = [len(z) for z in z_eos]
    max_len = max(z_lengths)
    z_padded = [pad_seq(z, max_len) for z in z_eos]
    z_padded = [lang.variableFromSymbols(z, add_eos=False).unsqueeze(0) for z in z_padded]
    z_padded = torch.cat(z_padded,dim=0)
    if USE_CUDA:
        z_padded = z_padded.cuda()
    return z_padded,z_lengths

def build_sample(x_support,y_support,x_query,y_query,input_lang,output_lang,myhash,grammar=''):
    # Build an episode from input/output examples
    # 
    # Input
    #  x_support [length ns list of lists] : input sequences (each a python list of words/symbols)
    #  y_support [length ns list of lists] : output sequences (each a python list of words/symbols)
    #  x_query [length nq list of lists] : input sequences (each a python list of words/symbols)
    #  x_query [length nq list of lists] : output sequences (each a python list of words/symbols)
    #  input_lang: Language object for input sequences (see Language)
    #  output_lang: Language object for output sequences
    #  myhash : unique string identifier for this episode (should be order invariant for examples)
    #  grammar : (optional) grammar object
    #
    # Output
    #  sample : dict that stores episode information
    sample = {}

    # store input and output sequences
    sample['identifier'] = myhash
    sample['xs'] = x_support 
    sample['ys'] = y_support
    sample['xq'] = x_query
    sample['yq'] = y_query
    sample['grammar'] = grammar
    
    # convert strings to indices, pad, and create tensors ready for input to network
    sample['xs_padded'],sample['xs_lengths'] = build_padded_var(x_support,input_lang) # (ns x max_length)
    sample['ys_padded'],sample['ys_lengths'] = build_padded_var(y_support,output_lang) # (ns x max_length)
    sample['xq_padded'],sample['xq_lengths'] = build_padded_var(x_query,input_lang) # (nq x max_length)
    sample['yq_padded'],sample['yq_lengths'] = build_padded_var(y_query,output_lang) # (nq x max_length)
    return sample

def extract(include,arr):
    # Create a new list only using the included (boolean) elements of arr 
    #
    # Input
    #  include : [n len] boolean array
    #  arr [ n length array]
    assert len(include)==len(arr)
    return [a for idx,a in enumerate(arr) if include[idx]]

def evaluation_battery(sample_eval_list, encoder, decoder, input_lang, output_lang, max_length, verbose=False):
    # Evaluate a list of episodes
    #
    # Input 
    #   sample_eval_list : list of evaluation sets to iterate through
    #   ...
    #   input_lang: Language object for input sequences
    #   output_lang: Language object for output sequences
    #   max_length : maximum length of a generated sequence
    #   verbose : print outcome or not?
    # 
    # Output
    #   (acc_novel, acc_autoencoder) average accuracy for novel items in query set, and support items in query set
    list_acc_val_novel = []
    list_acc_val_autoencoder = []
    for idx,sample in enumerate(sample_eval_list):
        acc_val_novel, acc_val_autoencoder, yq_predict, in_support, all_attention_by_query, memory_attn_steps = evaluate(sample, encoder, decoder, input_lang, output_lang, max_length)
        list_acc_val_novel.append(acc_val_novel)
        list_acc_val_autoencoder.append(acc_val_autoencoder)
        if verbose:
            print('')
            print('Evaluation episode ' + str(idx))
            if sample['grammar']:
                print("")
                print(sample['grammar'])
            print('  support items: ')
            display_input_output(sample['xs'],sample['ys'],sample['ys'])
            print('  retrieval items; ' + str(round(acc_val_autoencoder,3)) + '% correct')
            display_input_output(extract(in_support,sample['xq']),extract(in_support,yq_predict),extract(in_support,sample['yq']))
            print('  generalization items; ' + str(round(acc_val_novel,3)) + '% correct')
            display_input_output(extract(np.logical_not(in_support),sample['xq']),extract(np.logical_not(in_support),yq_predict),extract(np.logical_not(in_support),sample['yq']))
    return np.mean(list_acc_val_novel), np.mean(list_acc_val_autoencoder)

def evaluate(sample, encoder, decoder, input_lang, output_lang, max_length):
    # Evaluate an episode
    # 
    # Input
    #   sample : [dict] generated validation episode, produced by "build_sample"
    #   ...
    #   input_lang: Language object for input sequences
    #   output_lang: Language object for output sequences
    #   max_length : maximum length of generated output sequence
    #
    # Output
    #   acc_novel : accuracy (percent correct) on novel items in query set
    #   acc_autoencoder : accuracy (percent correct) on reconstructing support items in query set
    #   yq_predict : list of predicted output sequences for all items
    #   is_support : [n x 1 bool] indicates for each query item whether it is in the support set
    #   all_attn_by_time : list (over time step) of batch_size x max_input_length tensors
    #   memory_attn_steps : attention over support items at every step for each query [max_xq_length x nq x ns]
    encoder.eval()
    decoder.eval()

    # Run words through encoder
    encoder_embedding, dict_encoder = encoder(sample)
    encoder_embedding_steps = dict_encoder['embed_by_step']
    memory_attn_steps = dict_encoder['attn_by_step']
    
    # Prepare input and output variables
    nq = len(sample['yq'])
    decoder_input = torch.tensor([output_lang.symbol2index[SOS_token]]*nq) # nq length tensor
    decoder_hidden = decoder.initHidden(encoder_embedding)

    # Store output words and attention states
    decoded_words = []
    
    # Run through decoder
    all_decoder_outputs = np.zeros((nq, max_length), dtype=int)
    all_attn_by_time = [] # list (over time step) of batch_size x max_input_length tensors
    if USE_CUDA:
        decoder_input = decoder_input.cuda()    
    for t in range(max_length):
        if type(decoder) is AttnDecoderRNN:
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_embedding_steps)
            all_attn_by_time.append(attn_weights)
        elif type(decoder) is DecoderRNN:        
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        else:
            assert False
        
        # Choose top symbol from output
        topv, topi = decoder_output.cpu().data.topk(1)
        decoder_input = topi.view(-1)
        if USE_CUDA:
            decoder_input = decoder_input.cuda()
        all_decoder_outputs[:,t] = topi.numpy().flatten()

    # get predictions
    in_support = np.array([x in sample['xs'] for x in sample['xq']])
    yq_predict = []
    for q in range(nq):
        myseq = output_lang.symbolsFromVector(all_decoder_outputs[q,:])
        yq_predict.append(myseq)
    
    # compute accuracy
    v_acc = np.zeros(nq)
    for q in range(nq):
        v_acc[q] = yq_predict[q] == sample['yq'][q]
    acc_autoencoder = np.mean(v_acc[in_support])*100.
    acc_novel = np.mean(v_acc[np.logical_not(in_support)])*100.
    return acc_novel, acc_autoencoder, yq_predict, in_support, all_attn_by_time, memory_attn_steps

def train(sample, encoder, decoder, encoder_optimizer, decoder_optimizer, input_lang, output_lang):
    # Update the model for a single training episode
    # 
    # Input
    #   sample : [dict] generated training episode, produced by "build_sample"
    #   ...
    #   input_lang: Language object for input sequences
    #   output_lang: Language object for output sequences
    #

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder.train()
    decoder.train()

    # Run words through encoder
    encoder_embedding, dict_encoder = encoder(sample)
    encoder_embedding_steps = dict_encoder['embed_by_step']
    
    # Prepare input and output variables
    nq = len(sample['yq']) # number of queries
    decoder_input = torch.tensor([output_lang.symbol2index[SOS_token]]*nq) # nq length tensor
    decoder_hidden = decoder.initHidden(encoder_embedding)
    target_batches = torch.transpose(sample['yq_padded'], 0, 1) # (max_length x nq tensor) ... batch targets with padding    
    target_lengths = sample['yq_lengths']
    max_target_length = max(target_lengths)
    all_decoder_outputs = torch.zeros(max_target_length, nq, decoder.output_size)
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        target_batches = target_batches.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()
    
    # Run through decoder one time step at a time
    for t in range(max_target_length):
        if type(decoder) is AttnDecoderRNN:
            decoder_output, decoder_hidden, attn_by_query = decoder(decoder_input, decoder_hidden, encoder_embedding_steps)
        elif type(decoder) is DecoderRNN:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        else:
            assert False
        all_decoder_outputs[t] = decoder_output # max_len x nq x output_size
        decoder_input = target_batches[t]

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        torch.transpose(all_decoder_outputs, 0, 1).contiguous(), # -> nq x max_length
        torch.transpose(target_batches, 0, 1).contiguous(), # nq x max_length
        target_lengths
    )

    # gradient update
    loss.backward()
    encoder_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    decoder_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    if encoder_norm > clip or decoder_norm > clip:
        print("Gradient clipped:")
        print("  Encoder norm: " + str(encoder_norm))
        print("  Decoder norm: " + str(decoder_norm))
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.cpu().item()

def display_input_output(input_patterns,output_patterns,target_patterns):
    # Verbose analysis of performance on query items
    # 
    # Input
    #   input_patterns : list of input sequences (each in list form)
    #   output_patterns : list of output sequences, which are actual outputs (each in list form)
    #   target_patterns : list of targets
    nq = len(input_patterns)
    if nq == 0:
        print('     no patterns')
        return
    for q in range(nq):
        assert isinstance(input_patterns[q],list)
        assert isinstance(output_patterns[q],list)
        is_correct = output_patterns[q] == target_patterns[q]        
        print('     ',end='')
        print(' '.join(input_patterns[q]),end='')
        print(' -> ',end='')
        print(' '.join(output_patterns[q]),end='')
        if not is_correct:
            print(' (** target: ',end='')
            print(' '.join(target_patterns[q]),end='')
            print(')',end='')
        print('')

def get_episode_generator(episode_type):
    #  Returns function that generates episodes,
    #   and language class for the input and output language
    #
    # Input
    #  episode_type : string specifying type of episode
    #
    # Output
    #  generate_episode: function handle for generating episodes
    #  input_lang: Language object for input sequence
    #  output_lang: Language object for output sequence
    input_symbols_list_default = ['dax', 'lug', 'wif', 'zup', 'fep', 'blicket', 'kiki', 'tufa', 'gazzer']
    output_symbols_list_default = ['RED', 'YELLOW', 'GREEN', 'BLUE', 'PURPLE', 'PINK']
    input_lang = Lang(input_symbols_list_default)
    output_lang = Lang(output_symbols_list_default)
    if episode_type == 'ME': # NeurIPS Exp 1 : Mutual exclusivity
        input_lang = Lang(input_symbols_list_default[:4])
        output_lang = Lang(output_symbols_list_default[:4])
        generate_episode_train = lambda tabu_episodes : generate_ME(nquery=20,nprims=len(input_lang.symbols),input_lang=input_lang,output_lang=output_lang,tabu_list=tabu_episodes)
        generate_episode_test = generate_episode_train
    elif episode_type == 'scan_prim_permutation': # NeurIPS Exp 2 : Adding a new primitive through permutation meta-training
        scan_all = ge.load_scan_file('all','train')
        scan_all_var = ge.load_scan_var('all','train')
        input_symbols_scan = get_unique_words([c[0] for c in scan_all])
        output_symbols_scan = get_unique_words([c[1] for c in scan_all])
        input_lang = Lang(input_symbols_scan)
        output_lang = Lang(output_symbols_scan)
        generate_episode_train = lambda tabu_episodes : generate_prim_permutation(shuffle=True, nsupport=20, nquery=20, input_lang=input_lang, output_lang=output_lang, scan_var_tuples=scan_all_var, nextra=0, tabu_list=tabu_episodes)
        generate_episode_test = lambda tabu_episodes : generate_prim_permutation(shuffle=False, nsupport=20, nquery=20, input_lang=input_lang, output_lang=output_lang, scan_var_tuples=scan_all_var, nextra=0, tabu_list=tabu_episodes)
    elif episode_type == 'scan_prim_augmentation': # NeurIPS Exp 3 : Adding a new primitive through augmentation meta-training
        nextra_prims = 20
        scan_all = ge.load_scan_file('all','train')
        scan_all_var = ge.load_scan_var('all','train')
        input_symbols_scan = get_unique_words([c[0] for c in scan_all]  + [str(i) for i in range(1,nextra_prims+1)])
        output_symbols_scan = get_unique_words([c[1] for c in scan_all] + ['I_' + str(i) for i in range(1,nextra_prims+1)])
        input_lang = Lang(input_symbols_scan)
        output_lang = Lang(output_symbols_scan)
        generate_episode_train = lambda tabu_episodes : generate_prim_augmentation(shuffle=True, nextra=nextra_prims, nsupport=20, nquery=20, input_lang=input_lang, output_lang=output_lang, scan_var_tuples=scan_all_var, tabu_list=tabu_episodes)
        generate_episode_test = lambda tabu_episodes : generate_prim_augmentation(shuffle=False, nextra=0, nsupport=20, nquery=20, input_lang=input_lang, output_lang=output_lang, scan_var_tuples=scan_all_var, tabu_list=tabu_episodes)                    
    elif episode_type == 'scan_around_right': # NeurIPS Exp 4 : Combining familiar concepts through meta-training
        nextra_prims = 2
        scan_all = ge.load_scan_file('all','train')
        scan_all_var = ge.load_scan_dir_var('all','train')        
        input_symbols_scan = get_unique_words([c[0] for c in scan_all]  + [str(i) for i in range(1,nextra_prims+1)])
        output_symbols_scan = get_unique_words([c[1] for c in scan_all] + ['I_' + str(i) for i in range(1,nextra_prims+1)])
        input_lang = Lang(input_symbols_scan)
        output_lang = Lang(output_symbols_scan)
        generate_episode_train = lambda tabu_episodes : generate_right_augmentation(shuffle=True, nextra=nextra_prims, nsupport=20, nquery=20, input_lang=input_lang, output_lang=output_lang, scan_var_tuples=scan_all_var, tabu_list=tabu_episodes)
        generate_episode_test = lambda tabu_episodes : generate_right_augmentation(shuffle=False, nextra=0, nsupport=20, nquery=20, input_lang=input_lang, output_lang=output_lang, scan_var_tuples=scan_all_var, tabu_list=tabu_episodes)    
    elif episode_type == 'scan_length': # NeurIPS Exp 5 : Generalizing to longer instructions through meta-training
        nextra_prims = 20 # number of additional primitives to augment the episodes with
        support_threshold = 12 # items with action length less than this belong in the support, 
                               # and greater than or equal to this length belong in the query
        scan_length_train = ge.load_scan_file('length','train')
        scan_length_test = ge.load_scan_file('length','test')
        scan_all = scan_length_train+scan_length_test
        scan_length_train_var = ge.load_scan_var('length','train')
        scan_length_test_var = ge.load_scan_var('length','test')        
        input_symbols_scan = get_unique_words([c[0] for c in scan_all]  + [str(i) for i in range(1,nextra_prims+1)])
        output_symbols_scan = get_unique_words([c[1] for c in scan_all] + ['I_' + str(i) for i in range(1,nextra_prims+1)])
        input_lang = Lang(input_symbols_scan)
        output_lang = Lang(output_symbols_scan)
        scan_length_support_var = [pair for pair in scan_length_train_var if len(pair[1].split(' ')) < support_threshold] # partition based on number of output actions
        scan_length_query_var = [pair for pair in scan_length_train_var if len(pair[1].split(' ')) >= support_threshold] # long sequences
        generate_episode_train = lambda tabu_episodes : generate_length(shuffle=True, nextra=nextra_prims, nsupport=100, nquery=20, input_lang=input_lang, output_lang=output_lang,
                                                            scan_tuples_support_variable=scan_length_support_var, scan_tuples_query_variable=scan_length_query_var, tabu_list=tabu_episodes)
        generate_episode_test = lambda tabu_episodes : generate_length(shuffle=False, nextra=0, nsupport=100, nquery=20, input_lang=input_lang, output_lang=output_lang,
                                                            scan_tuples_support_variable=scan_length_train_var, scan_tuples_query_variable=scan_length_test_var, tabu_list=tabu_episodes)
    else:
        raise Exception("episode_type is not valid" )
    return generate_episode_train, generate_episode_test, input_lang, output_lang

def generate_ME(nquery,nprims,input_lang,output_lang,maxlen=6,tabu_list=[]):
    # Sample mutual exclusivity episode
    #
    # Input
    #  nquery : number of query examples
    #  nprims : number of unique primitives (support set includes all but one)
    #  maxlen : maximum length of a sequence in the episode
    #  ...
    #  tabu_list : identifiers of episodes we should not produce
    #

    input_symbols = deepcopy(input_lang.symbols)
    output_symbols = deepcopy(output_lang.symbols)     
    assert(nprims == len(input_symbols))
    count = 0
    while True:
        random.shuffle(input_symbols)
        random.shuffle(output_symbols)
        D_str = '\n'.join([input_symbols[idx] + ' -> ' + output_symbols[idx] for idx in range(nprims)])   
        identifier = make_hashable(D_str)        
        D_support,D_query = ge.sample_ME_concat_data(nquery=nquery,input_symbols=input_symbols,output_symbols=output_symbols,maxlen=maxlen,inc_support_in_query=use_resconstruct_loss)
        if identifier not in tabu_list:
            break
        count += 1
        if count > max_try_novel:
            raise Exception('We were unable to generate an episode that is not on the tabu list')        
    x_support = [d[0].split(' ') for d in D_support]
    y_support = [d[1].split(' ') for d in D_support]
    x_query = [d[0].split(' ') for d in D_query]
    y_query = [d[1].split(' ') for d in D_query]
    return build_sample(x_support,y_support,x_query,y_query,input_lang,output_lang,identifier)

def generate_prim_permutation(shuffle,nsupport,nquery,input_lang,output_lang,scan_var_tuples,nextra,tabu_list=[]):
    # Generate a SCAN episode with primitive permutation.
    #  The tabu list identifier is based on the permutation of primitive inputs to primitive actions.
    #
    # Input
    #  shuffle: permute how the input primitives map to the output actions? (true/false)
    #  scan_var_tuples : scan input/output sequences with placeholder replacement
    #  nextra: number of abstract input/output primitives to add to the set of possibilities
    #
    count = 0
    while True:
        D_support, D_query, D_primitive = ge.sample_augment_scan(nsupport,nquery,scan_var_tuples,shuffle,nextra,inc_support_in_query=use_resconstruct_loss)
        D_str = '\n'.join([s[0] + ' -> ' + s[1] for s in D_primitive])
        identifier = make_hashable(D_str)
        if not shuffle: # ignore tabu list if we aren't shuffling primitive assignments
            break
        if identifier not in tabu_list:
            break
        count += 1
        if count > max_try_novel:
            raise Exception('We were unable to generate an episode that is not on the tabu list')
    x_support = [d[0].split(' ') for d in D_support]
    y_support = [d[1].split(' ') for d in D_support]
    x_query = [d[0].split(' ') for d in D_query]
    y_query = [d[1].split(' ') for d in D_query]
    return build_sample(x_support,y_support,x_query,y_query,input_lang,output_lang,identifier)    

def generate_prim_augmentation(shuffle,nsupport,nquery,input_lang,output_lang,scan_var_tuples,nextra,tabu_list=[]):
    # Generate a SCAN episode with primitive augmentation,
    #  The tabu list identifier is only determined based on the assignment of the "jump" primitive 
    #
    # Input
    #  shuffle: permute how the input primitives map to the output actions? (true/false)
    #  scan_var_tuples : scan input/output patterns with placeholder replacement
    #  nextra: number of abstract input/output primitives to add to the set of possibilities
    #
    special_prim = 'jump'
    count = 0
    while True:
        D_support, D_query, D_primitive = ge.sample_augment_scan(nsupport,nquery,scan_var_tuples,shuffle,nextra,inc_support_in_query=use_resconstruct_loss)
        input_prim_list = [s[0] for s in D_primitive]
        try:
            index_prim = input_prim_list.index(special_prim)
            D_str = D_primitive[index_prim][0] + ' -> ' + D_primitive[index_prim][1]
        except ValueError:
            D_str = 'no jump'    
        identifier = D_str
        if not shuffle: # ignore tabu list if we aren't shuffling primitive assignments
            break
        if identifier not in tabu_list:
            break
        count += 1
        if count > max_try_novel:
            raise Exception('We were unable to generate an episode that is not on the tabu list')
    x_support = [d[0].split(' ') for d in D_support]
    y_support = [d[1].split(' ') for d in D_support]
    x_query = [d[0].split(' ') for d in D_query]
    y_query = [d[1].split(' ') for d in D_query]
    return build_sample(x_support,y_support,x_query,y_query,input_lang,output_lang,identifier)

def generate_right_augmentation(shuffle,nsupport,nquery,input_lang,output_lang,scan_var_tuples,nextra,tabu_list=[]):
    # Generate a SCAN episode with primitive augmentation,
    #  The tabu list is only determined based on the assignment of the "right" primitive
    #
    # Input
    #  shuffle: permute how the input primitives map to the output actions? (true/false)
    #  scan_var_tuples : scan input/output patterns with placeholder replacement
    #  nextra: number of abstract input/output primitives to add to the set of possibilities
    #
    special_prim = 'right'
    count = 0
    while True:
        D_support, D_query, D_angles = ge.sample_augment_direction_scan(nsupport,nquery,scan_var_tuples,shuffle,nextra,inc_support_in_query=use_resconstruct_loss)
        input_angle_list = [s[0] for s in D_angles]
        try:
            index_prim = input_angle_list.index(special_prim)
            D_str = D_angles[index_prim][0] + ' -> ' + D_angles[index_prim][1]
        except ValueError:
            D_str = 'no right'    
        identifier = D_str
        if not shuffle: # ignore tabu list if we aren't shuffling primitive assignments
            break
        if identifier not in tabu_list:
            break
        count += 1
        if count > max_try_novel:
            raise Exception('We were unable to generate an episode that is not on the tabu list')
    x_support = [d[0].split(' ') for d in D_support]
    y_support = [d[1].split(' ') for d in D_support]
    x_query = [d[0].split(' ') for d in D_query]
    y_query = [d[1].split(' ') for d in D_query]
    return build_sample(x_support,y_support,x_query,y_query,input_lang,output_lang,identifier)

def generate_length(shuffle,nsupport,nquery,input_lang,output_lang,scan_tuples_support_variable,scan_tuples_query_variable,nextra,tabu_list=[]):
    # ** This episode allows different sets of input/output patterns for the support and query **
    # Generate a SCAN episode with primitive augmentation. 
    #  The tabu list is based on the assignment of all of the primitive inputs to primitive actions.
    #
    # Input
    #  shuffle: permute how the input primitives map to the output actions? (true/false)
    #  scan_tuples_support_variable : scan input/output patterns with placeholder replacement
    #  scan_tuples_query_variable : scan input/output patterns with placeholder replacement
    #  nextra: number of abstract input/output primitives to add to the set of possibilities
    #
    count = 0
    while True:
        D_support, D_query, D_primitive = ge.sample_augment_scan_separate(nsupport,nquery,scan_tuples_support_variable,scan_tuples_query_variable,shuffle,nextra,inc_support_in_query=use_resconstruct_loss)
        D_str = '\n'.join([s[0] + ' -> ' + s[1] for s in D_primitive])
        identifier = make_hashable(D_str)
        if not shuffle: # ignore tabu list if we aren't shuffling primitive assignments
            break
        if identifier not in tabu_list:
            break
        count += 1
        if count > max_try_novel:
            raise Exception('We were unable to generate an episode that is not on the tabu list')
    x_support = [d[0].split(' ') for d in D_support]
    y_support = [d[1].split(' ') for d in D_support]
    x_query = [d[0].split(' ') for d in D_query]
    y_query = [d[1].split(' ') for d in D_query]
    return build_sample(x_support,y_support,x_query,y_query,input_lang,output_lang,identifier)
        
if __name__ == "__main__":

    # Training parameters
    num_episodes_val = 5 # number of episodes to use as validation throughout learning
    clip = 50.0 # clip gradients with larger magnitude than this
    max_try_novel = 100 # number of attempts to find a novel episode (not in tabu list) before throwing an error
    
    # Adjustable parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=10000, help='number of episodes for training')
    parser.add_argument('--lr', type=float, default=0.001, help='ADAM learning rate')
    parser.add_argument('--lr_decay_schedule', type=int, default=5000, help='decrease learning rate by 1e-1 after this many episodes')
    parser.add_argument('--nlayers', type=int, default=2, help='number of layers in the LSTM')
    parser.add_argument('--max_length_eval', type=int, default=50, help='maximum generated sequence length when evaluating the network')
    parser.add_argument('--emb_size', type=int, default=200, help='size of sequence embedding (also, nhidden for encoder and decoder LSTMs)')
    parser.add_argument('--dropout', type=float, default=0.5, help=' dropout applied to embeddings and LSTMs')
    parser.add_argument('--fn_out_model', type=str, default='', help='filename for saving the model')
    parser.add_argument('--dir_model', type=str, default='out_models', help='directory for saving model files')
    parser.add_argument('--episode_type', type=str, default='ME', help='what type of episodes do we want')
    parser.add_argument('--disable_memory', action='store_true', help='Disable external memory, ignore support set, and use simple RNN encoder')
    parser.add_argument('--disable_attention', action='store_true', help='Disable the decoder attention')
    parser.add_argument('--disable_recon_loss', action='store_true', help='Disable reconstruction loss, where support items are included also as query items')
    parser.add_argument('--gpu', type=int, default=0, help='set which GPU we want to use')
    args = parser.parse_args()
    fn_out_model = args.fn_out_model
    episode_type = args.episode_type
    dir_model = args.dir_model
    gpu_num = args.gpu
    emb_size = args.emb_size
    nlayers = args.nlayers
    num_episodes = args.num_episodes
    dropout_p = args.dropout
    adam_learning_rate = args.lr
    lr_decay_schedule = args.lr_decay_schedule
    max_length_eval = args.max_length_eval
    disable_memory = args.disable_memory
    disable_recon_loss = args.disable_recon_loss
    use_resconstruct_loss = not disable_recon_loss
    use_attention = not args.disable_attention

    if fn_out_model=='':
        fn_out_model = 'net_'  + episode_type + '.tar'
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)
    fn_out_model = os.path.join(dir_model, fn_out_model)

    if not os.path.isfile(fn_out_model):
        print("Training a new network...")
        print("  Episode type is " + episode_type)        
        generate_episode_train, generate_episode_test, input_lang,output_lang = get_episode_generator(episode_type)
        if USE_CUDA:
            torch.cuda.set_device(gpu_num)
            print('  Training on GPU ' + str(torch.cuda.current_device()), end='')
        else:
            print('  Training on CPU', end='')
        print(' for ' + str(num_episodes) + ' episodes')
        input_size = input_lang.n_symbols
        output_size = output_lang.n_symbols
        
        if disable_memory:
            encoder = WrapperEncoderRNN(emb_size, input_size, output_size, nlayers, dropout_p)
        else:
            encoder = MetaNetRNN(emb_size, input_size, output_size, nlayers, dropout_p)
        if use_attention:
            decoder = AttnDecoderRNN(emb_size, output_size, nlayers, dropout_p)
        else:
            decoder = DecoderRNN(emb_size, output_size, nlayers, dropout_p)
        if USE_CUDA:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
        criterion = nn.NLLLoss()
        print('  Set learning rate to ' + str(adam_learning_rate))
        encoder_optimizer = optim.Adam(encoder.parameters(),lr=adam_learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(),lr=adam_learning_rate)
        print("")
        print("Architecture options...")
        print(" Decoder attention is USED" ) if use_attention else print(" Decoder attention is NOT used")
        print(" External memory is USED" ) if not disable_memory else print(" External memory is NOT used")
        print(" Reconstruction loss is USED" ) if not disable_recon_loss else print(" Reconstruction loss is NOT used")
        print("")
        describe_model(encoder)
        describe_model(decoder)

        # create validation episodes
        tabu_episodes = set([])
        samples_val = []
        for i in range(num_episodes_val):
            sample = generate_episode_test(tabu_episodes)
            samples_val.append(sample)
            tabu_episodes = tabu_update(tabu_episodes,sample['identifier'])

        # train over a set of random episodes
        avg_train_loss = 0.
        counter = 0 # used to count updates since the loss was last reported
        start = time.time()
        for episode in range(1,num_episodes+1):

            # generate a random episode
            sample = generate_episode_train(tabu_episodes)
            
            # batch updates (where batch includes the entire support set)
            train_loss = train(sample, encoder, decoder, encoder_optimizer, decoder_optimizer, input_lang, output_lang)
            avg_train_loss += train_loss
            counter += 1

            if episode == 1 or episode % 100 == 0 or episode == num_episodes:
                acc_val_gen, acc_val_retrieval = evaluation_battery(samples_val, encoder, decoder, input_lang, output_lang, max_length_eval)
                print('{:s} ({:d} {:.0f}% finished) TrainLoss: {:.4f}, ValAccRetrieval: {:.1f}, ValAccGeneralize: {:.1f}'.format(timeSince(start, float(episode) / float(num_episodes)),
                                         episode, float(episode) / float(num_episodes) * 100., avg_train_loss/counter, acc_val_retrieval, acc_val_gen))
                avg_train_loss = 0.
                counter = 0
                if episode % 1000 == 0 or episode == num_episodes:
                    state = {'encoder_state_dict': encoder.state_dict(),
                                'decoder_state_dict': decoder.state_dict(),
                                'input_lang': input_lang,
                                'output_lang': output_lang,
                                'episodes_validation': samples_val,
                                'episode_type': episode_type,
                                'emb_size':emb_size,
                                'dropout':dropout_p,
                                'nlayers':nlayers,
                                'episode':episode,
                                'disable_memory':disable_memory,
                                'disable_recon_loss':disable_recon_loss,
                                'use_attention':use_attention,
                                'max_length_eval':max_length_eval,
                                'num_episodes':num_episodes,
                                'args':args}
                    print('Saving model as: ' + fn_out_model)
                    torch.save(state, fn_out_model)

            # decay learning rate according to schedule
            if episode % lr_decay_schedule == 0:
                adam_learning_rate = adam_learning_rate * 0.1
                encoder_optimizer = optim.Adam(encoder.parameters(),lr=adam_learning_rate)
                decoder_optimizer = optim.Adam(decoder.parameters(),lr=adam_learning_rate)
                print('Set learning rate to ' + str(adam_learning_rate))

        print('Training complete')
        acc_val_gen, acc_val_retrieval = evaluation_battery(samples_val, encoder, decoder, input_lang, output_lang, max_length_eval, verbose=False)
        print('Acc Retrieval (val): ' + str(round(acc_val_retrieval,1)))
        print('Acc Generalize (val): ' + str(round(acc_val_gen,1)))
    else: # evaluate model if filename already exists
        USE_CUDA = False
        print('Results file already exists. Loading file and evaluating...')
        print('Loading model: ' + fn_out_model)        
        checkpoint = torch.load(fn_out_model, map_location='cpu') # evaluate model on CPU
        if 'episode' in checkpoint: print(' Loading epoch ' + str(checkpoint['episode']) + ' of ' + str(checkpoint['num_episodes']))
        input_lang = checkpoint['input_lang']
        output_lang = checkpoint['output_lang']
        emb_size = checkpoint['emb_size']
        nlayers = checkpoint['nlayers']
        dropout_p = checkpoint['dropout']
        input_size = input_lang.n_symbols
        output_size = output_lang.n_symbols
        samples_val = checkpoint['episodes_validation']
        disable_memory = checkpoint['disable_memory']
        max_length_eval = checkpoint['max_length_eval']
        if 'args' not in checkpoint or 'disable_attention' not in checkpoint['args']:
            use_attention = True
        else:
            args = checkpoint['args']
            use_attention = not args.disable_attention
        if disable_memory:
            encoder = WrapperEncoderRNN(emb_size, input_size, output_size, nlayers, dropout_p)
        else:
            encoder = MetaNetRNN(emb_size, input_size, output_size, nlayers, dropout_p)        
        if use_attention:
            decoder = AttnDecoderRNN(emb_size, output_size, nlayers, dropout_p)
        else:
            decoder = DecoderRNN(emb_size, output_size, nlayers, dropout_p)
        if USE_CUDA:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        describe_model(encoder)
        describe_model(decoder)

        generate_episode_train, generate_episode_test, input_lang,output_lang = get_episode_generator(episode_type)
        acc_train_gen, acc_train_retrieval = evaluation_battery([generate_episode_train([]) for _ in range(5)], encoder, decoder, input_lang, output_lang, max_length_eval, verbose=False)
        print('Acc Retrieval (train): ' + str(round(acc_train_retrieval,1)))
        print('Acc Generalize (train): ' + str(round(acc_train_gen,1)))

        acc_val_gen, acc_val_retrieval = evaluation_battery(samples_val, encoder, decoder, input_lang, output_lang, max_length_eval, verbose=True)
        print('Acc Retrieval (val): ' + str(round(acc_val_retrieval,1)))
        print('Acc Generalize (val): ' + str(round(acc_val_gen,1)))