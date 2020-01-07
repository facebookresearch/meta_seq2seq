# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import argparse
import random
import os
from contextlib import redirect_stdout
from copy import deepcopy, copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import MetaNetRNN, AttnDecoderRNN, DecoderRNN, describe_model, WrapperEncoderRNN
import generate_episode as ge
from train import evaluation_battery, Lang, build_sample, evaluate, display_input_output

# --
# Evaluate a trained meta seq2seq model on a SCAN split ("add jump", "around right", length", etc.)
# 
# See Main for example of how to run...
# --

USE_CUDA = False

def scan_evaluation_prim_only(mytype,split,input_lang,output_lang):
    # Load an entire SCAN split as the query set.
    #   Use the isolated primitives as the support set
    #
    # Input
    #  mytype : type of SCAN experiment
    #  split : 'train' or 'test'
    #  ... other inputs are language objects
    D_query = ge.load_scan_file(mytype,split)
    _, _, D_primitive = ge.sample_augment_scan(0,0,[],shuffle=False,inc_support_in_query=False)
    D_support = D_primitive # support set only includes the primitive mappings...
    random.shuffle(D_support)
    x_support = [d[0].split(' ') for d in D_support]
    y_support = [d[1].split(' ') for d in D_support]
    x_query = [d[0].split(' ') for d in D_query]
    y_query = [d[1].split(' ') for d in D_query]
    return build_sample(x_support,y_support,x_query,y_query,input_lang,output_lang,'')

def scan_evaluation_dir_only(mytype,split,input_lang,output_lang):
    # Load an entire SCAN pattern file as the query set
    #  Just use the isolated directions as the support set
    #
    # Input
    #  mytype : type of SCAN experiment
    #  split : 'train' or 'test'
    #  ... other inputs are language objects
    D_query = ge.load_scan_file(mytype,split)
    D_support = [('turn left', 'I_TURN_LEFT'), ('turn right','I_TURN_RIGHT')]
    random.shuffle(D_support)
    x_support = [d[0].split(' ') for d in D_support]
    y_support = [d[1].split(' ') for d in D_support]
    x_query = [d[0].split(' ') for d in D_query]
    y_query = [d[1].split(' ') for d in D_query]
    return build_sample(x_support,y_support,x_query,y_query,input_lang,output_lang,'')

def scan_evaluation_val_support(mytype,split,input_lang,output_lang,samples_val):
    # Use the pre-generated in the validation episodes as the support set.
    #  Replace the validation episodes' query sets as the rest of the SCAN split (e.g., the entire length test set)
    #  
    # Input
    #  mytype : type of SCAN experiment
    #  split : 'train' or 'test'
    #  ... other inputs are language objects
    #  samples_val : list of pre-generated validation episodes
    D_query = ge.load_scan_file(mytype,split) # e.g., we can load in the entire "length" test set
    x_query = [d[0].split(' ') for d in D_query]
    y_query = [d[1].split(' ') for d in D_query]
    for idx in range(len(samples_val)):
        samples = samples_val[idx]
        samples_val[idx] = build_sample(samples['xs'],samples['ys'],deepcopy(x_query),deepcopy(y_query),input_lang,output_lang,'')
    return samples_val

def eval_network(fn_in_model):
    # Input
    #  fn_in_model : filename of saved model
    #
    # Create filename for output
    fn_out_res = fn_in_model
    fn_out_res = fn_out_res.replace('.tar','.txt')
    fn_out_res_test = fn_out_res.replace('/net_','/res_test_')

    # Load and evaluate the network in filename 'fn_in_model'
    assert(os.path.isfile(fn_in_model))
    print('  Checkpoint found...')
    print('  Processing model: ' + fn_in_model)
    print('  Writing to file: ' + fn_out_res_test)        
    checkpoint = torch.load(fn_in_model, map_location='cpu') # evaluate model on CPU    
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
   
    with open(fn_out_res_test, 'w') as f_test:
        with redirect_stdout(f_test):
            if 'episode' in checkpoint:
                print(' Loading epoch ' + str(checkpoint['episode']) + ' of ' + str(checkpoint['num_episodes']))
            describe_model(encoder)
            describe_model(decoder)
            if eval_type == 'val': 
                print('Evaluating VALIDATION performance on pre-generated validation set')
                acc_val_gen, acc_val_retrieval = evaluation_battery(samples_val, encoder, decoder, input_lang, output_lang, max_length_eval, verbose=True)
                print('Acc Retrieval (val): ' + str(round(acc_val_retrieval,1)))
                print('Acc Generalize (val): ' + str(round(acc_val_gen,1)))
            elif eval_type == 'addprim_jump':                    
                print('Evaluating TEST performance on SCAN addprim_jump')
                print('  ...support set is just the isolated primitives')
                mybatch = scan_evaluation_prim_only('addprim_jump','test',input_lang,output_lang)
                acc_val_gen, acc_val_retrieval = evaluation_battery([mybatch], encoder, decoder, input_lang, output_lang, max_length_eval, verbose=True)
            elif eval_type == 'length':                    
                print('Evaluating TEST performance on SCAN length')
                print('  ...over multiple support sets as contributed by the pre-generated validation set')
                samples_val = scan_evaluation_val_support('length','test',input_lang,output_lang,samples_val)
                acc_val_gen, acc_val_retrieval = evaluation_battery(samples_val, encoder, decoder, input_lang, output_lang, max_length_eval, verbose=True)
                print('Acc Retrieval (val): ' + str(round(acc_val_retrieval,1)))
                print('Acc Generalize (val): ' + str(round(acc_val_gen,1)))
            elif eval_type == 'template_around_right':
                print('Evaluating TEST performance on the SCAN around right')
                print(' ...with just direction mappings as support set')
                mybatch = scan_evaluation_dir_only('template_around_right','test',input_lang,output_lang)
                acc_val_gen, acc_val_retrieval = evaluation_battery([mybatch], encoder, decoder, input_lang, output_lang, max_length_eval, verbose=True)
            else:
                assert False

if __name__ == "__main__":

    # Adjustable parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn_in_model', type=str, default='', help='filename for saved model we want to load')
    parser.add_argument('--dir_model', type=str, default='out_models', help='directory of saved model file')
    parser.add_argument('--episode_type', type=str, default='ME', help='what type of episode the model was trained on')
    args = parser.parse_args()

    # File name for the model you want to load
    fn_in_model = args.dir_model + '/' + args.fn_in_model # e.g., 'out_models/net_scan_prim_permutation.tar'

    # Type of SCAN split you want to run
    episode_type = args.episode_type
    if (episode_type == 'scan_prim_permutation') or (episode_type == 'scan_prim_augmentation'):
        eval_type = 'addprim_jump'
    elif episode_type == 'scan_around_right':
        eval_type = 'template_around_right'
    elif episode_type == 'scan_length':
        eval_type = 'length'
    elif episode_type == 'ME':
        eval_type = 'val'
    else:
        assert False # invalid episode_type argument

    # Run evaluation    
    eval_network(fn_in_model)
        # Saves result as e.g.,'out_models/net_scan_prim_permutation.tar'