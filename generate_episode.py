# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
from copy import deepcopy

# --
# Generating episodes for meta learning.
# --

def sample_ME_concat_data(nquery,input_symbols,output_symbols,maxlen,maxntry=500,inc_support_in_query=False):
	# Sample ME episode based on current ordering of input/output symbols (already randomized)
	# 
	# Input
	#  nquery : number of query examples
	#  input_symbols : list of nprim input symbols (already permuted)
	#  output_symbols : list of nprim output symbols (already permuted)
	#  maxlen : maximum sequence length in query set
	#  inc_support_in_query : true/false, where true indicates that we include the "support loss" in paper (default=False)
	nprim = len(input_symbols)
	pairs = list(zip(input_symbols,output_symbols))

	# support set with all singleton primitives
	D_support = []
	for dat in pairs[:nprim-1]:
		D_support.append(dat)

	# query set with random concatenations
	D_query = set([])
	ntry = 0
	while len(D_query)<nquery:
		mylen = random.randint(2,maxlen)
		dat_list = [random.choice(pairs) for _ in range(mylen)]
		dat_in, dat_out = zip(*dat_list)		
		dat_in = ' '.join(dat_in)
		dat_out = ' '.join(dat_out)
		D_query.add((dat_in,dat_out))
		if ntry > maxntry:
			raise Exception('Maximum number of tries to generate valid dataset')	
	D_query = list(D_query)
	if inc_support_in_query:
		D_query += deepcopy(D_support)

	return D_support, D_query

def load_scan_file(mytype,split):
	# Load SCAN dataset from file
	#
	# Input
	#  mytype : type of SCAN experiment
	#  split : 'train' or 'test'
	#
	# Output
	#  commands : list of input/output strings (as tuples)
	assert mytype in ['simple','addprim_jump','length','addprim_turn_left','all','template_around_right','viz','examine']
	assert split in ['train','test']
	fn = 'data/tasks_' + split + '_' + mytype + '.txt'
	fid = open(fn,'r')
	lines = fid.readlines()
	fid.close()
	lines = [l.strip() for l in lines]
	lines = [l.lstrip('IN: ') for l in lines]
	commands = [l.split(' OUT: ') for l in lines]
	return commands

def sentence_replace_var(sentence,list_source,list_target):
	# Swap each source word in sentence with corresponding target word
	#
	# Input
	#  sentence: string of words
	#  list_source : length k list of words to be replaced
	#  list_target : length k list of words to replace source words
	#
	# Output
	#   sentence: new string of words
	assert(len(list_source)==len(list_target))
	for i in range(len(list_source)):
		sentence = sentence.replace(list_source[i],list_target[i])
	return sentence	

def load_scan_var(mytype,split):
	# Load SCAN tasks from file and replace action primitives (walk, look, run, jump) with variables
	#   Replace all input primitives with deterministic placeholders primitive1, primitive2, etc.
	#   Replace all output primitives with deterministic placeholders I_ACT_1, I_ACT_2, etc.
	#
	# Input
	#  mytype : type of SCAN experiment
	#  split : 'train' or 'test'
	#
	# Output
	#  commands : list of input/output strings (as tuples)
	scan_tuples = load_scan_file(mytype,split)
	scan_tuples_variable = deepcopy(scan_tuples)

	# original primitives
	scan_primitive_tuples = [('walk','I_WALK'),('look','I_LOOK'),('run','I_RUN'),('jump','I_JUMP')]
	nprim = len(scan_primitive_tuples)
	list_source_command = [p[0] for p in scan_primitive_tuples] # each input primitive in source
	list_source_output = [p[1] for p in scan_primitive_tuples] # each output primitive in source

	# replacement placeholder primitives
	scan_placeholder_tuples = [('primitive'+str(i),'I_ACT_'+str(i)) for i in range(1,nprim+1)]
	list_target_command = [p[0] for p in scan_placeholder_tuples] # each input primitive as target	
	list_target_output = [p[1] for p in scan_placeholder_tuples] # each output primitive as target

	# do replacement
	for i in range(len(scan_tuples_variable)):
		scan_tuples_variable[i][0] = sentence_replace_var(scan_tuples_variable[i][0], list_source_command, list_target_command)
		scan_tuples_variable[i][1] = sentence_replace_var(scan_tuples_variable[i][1], list_source_output, list_target_output)
	return scan_tuples_variable

def load_scan_dir_var(mytype,split):
	# Load SCAN tasks from file and replace turning primitives (right, left) with variables
	#   Replace all input primitives with deterministic placeholders primitive1, primitive2, etc.
	#   Replace all output primitives with deterministic placeholders I_ACT_1, I_ACT_2, etc.
	#
	# Input
	#  mytype : type of SCAN experiment
	#  split : 'train' or 'test'
	#
	# Output
	#  commands : list of input/output strings
	scan_tuples = load_scan_file(mytype,split)
	scan_tuples_variable = deepcopy(scan_tuples)

	# original primitives
	scan_primitive_tuples = [('right','I_TURN_RIGHT'),('left','I_TURN_LEFT')]
	nprim = len(scan_primitive_tuples)
	list_source_command = [p[0] for p in scan_primitive_tuples] # each input primitive in source
	list_source_output = [p[1] for p in scan_primitive_tuples] # each output primitive in source

	# replacement placeholder primitives
	scan_placeholder_tuples = [('primitive'+str(i),'I_ACT_'+str(i)) for i in range(1,nprim+1)]
	list_target_command = [p[0] for p in scan_placeholder_tuples] # each input primitive as target	
	list_target_output = [p[1] for p in scan_placeholder_tuples] # each output primitive as target

	# do replacement
	for i in range(len(scan_tuples_variable)):
		scan_tuples_variable[i][0] = sentence_replace_var(scan_tuples_variable[i][0], list_source_command, list_target_command)
		scan_tuples_variable[i][1] = sentence_replace_var(scan_tuples_variable[i][1], list_source_output, list_target_output)
	return scan_tuples_variable	

def sample_augment_scan(nsupport,nquery,scan_tuples_variable,shuffle,nextra=0,inc_support_in_query=False):
	# Both the query and the support set contain example input/output patterns
	#  Create an episode with shuffled input/output primitives (run, jump, walk, look), potentially with primitive augmentation
	#
	# Input
	#  nsupport : number of support items to sample
	#  nquery : number of query items to sample
	#  scan_tuples_variable : list of input/output tuples to draw from (in VARIABLE form, as generated by 'load_scan_var')
	#  shuffle : true/false; randomly shuffle the scan input/output primitives, or use semantic alignment?
	#  nextra : number of extra abstract input/output primitives to include
	#  inc_support_in_query : true/false; include the support set in the query set? (e.g., use support loss)
	#
	# Output
	#  D_support : list of support input/output pairs; in this case, it's the remapped primitives ONLY
	#  D_query : list of query input/output pairs
	#  D_primitive : list of primitive input/output pairs
	#
	# Sample query items from scan tuples (before variable replacement)
	# Distribute the patterns to query and support items
	scan_tuples_variable = deepcopy(scan_tuples_variable)
	random.shuffle(scan_tuples_variable)
	D_query = scan_tuples_variable[:nquery]
	D_support = scan_tuples_variable[nquery:nquery+nsupport]

	# Shuffle assignment of primitive commands to primitive actions
	nprim_replace = 4
	scan_primitive_tuples = [('walk','I_WALK'),('look','I_LOOK'),('run','I_RUN'),('jump','I_JUMP')]
	if nextra > 0:
		scan_primitives_extra = [(str(i),'I_'+str(i)) for i in range(1,nextra+1)]
		scan_primitive_tuples += scan_primitives_extra	
	unzip = list(zip(*scan_primitive_tuples))
	list_target_command = list(unzip[0])
	list_target_output = list(unzip[1])
	if shuffle: # shuffle assignment if desired
		random.shuffle(list_target_command)
		random.shuffle(list_target_output)
	list_target_command = list_target_command[:nprim_replace]
	list_target_output = list_target_output[:nprim_replace]

	# Replace placeholders with grounded commands and actions		
	scan_placeholder_tuples = [('primitive'+str(i),'I_ACT_'+str(i)) for i in range(1,nprim_replace+1)]
	list_source_command = [p[0] for p in scan_placeholder_tuples]
	list_source_output = [p[1] for p in scan_placeholder_tuples]
	for i in range(len(D_query)):
		D_query[i][0] = sentence_replace_var(D_query[i][0], list_source_command, list_target_command)
		D_query[i][1] = sentence_replace_var(D_query[i][1], list_source_output, list_target_output)
	for i in range(len(D_support)):
		D_support[i][0] = sentence_replace_var(D_support[i][0], list_source_command, list_target_command)
		D_support[i][1] = sentence_replace_var(D_support[i][1], list_source_output, list_target_output)
	
	D_primitive = list(zip(list_target_command,list_target_output))

	if inc_support_in_query:
		D_query += deepcopy(D_support)		
	return D_support, D_query, D_primitive

def sample_augment_direction_scan(nsupport,nquery,scan_tuples_variable,shuffle,nextra=0,inc_support_in_query=False):
	# Both the query and the support set contain example input/output patterns
	#  Create an episode with shuffled input/output directions (right, left, etc.), potentially with augmentation
	#
	# Input
	#  nsupport : number of support items to sample
	#  nquery : number of query items to sample
	#  scan_tuples_variable : list of input/output tuples to draw from (in VARIABLE form, as generated by 'load_scan_var')
	#  shuffle : true/false; randomly shuffle the scan input/output primitives, or use semantic alignment?
	#  nextra : number of extra abstract input/output primitives to include
	#  inc_support_in_query : true/false; include the support set in the query set?
	#
	# Output
	#  D_support : list of support input/output pairs; in this case, it's the remapped primitives ONLY
	#  D_query : list of query input/output pairs
	#  D_primitive : list of primitive input/output pairs
	#
	# Sample query items from scan tuples (before variable replacement)
	# Distribute the patterns to query and support items
	scan_tuples_variable = deepcopy(scan_tuples_variable)
	random.shuffle(scan_tuples_variable)
	D_query = scan_tuples_variable[:nquery]
	D_support = scan_tuples_variable[nquery:nquery+nsupport]

	# Shuffle assignment of primitive commands to primitive actions
	nprim_replace = 2
	scan_primitive_tuples = [('right','I_TURN_RIGHT'),('left','I_TURN_LEFT')]
	if nextra > 0:
		scan_primitives_extra = [(str(i),'I_'+str(i)) for i in range(1,nextra+1)]
		scan_primitive_tuples += scan_primitives_extra	
	unzip = list(zip(*scan_primitive_tuples))
	list_target_command = list(unzip[0])
	list_target_output = list(unzip[1])
	if shuffle: # shuffle assignment if desired
		random.shuffle(list_target_command)
		random.shuffle(list_target_output)
	list_target_command = list_target_command[:nprim_replace]
	list_target_output = list_target_output[:nprim_replace]

	# Replace placeholders with grounded commands and actions		
	scan_placeholder_tuples = [('primitive'+str(i),'I_ACT_'+str(i)) for i in range(1,nprim_replace+1)]
	list_source_command = [p[0] for p in scan_placeholder_tuples]
	list_source_output = [p[1] for p in scan_placeholder_tuples]
	for i in range(len(D_query)):
		D_query[i][0] = sentence_replace_var(D_query[i][0], list_source_command, list_target_command)
		D_query[i][1] = sentence_replace_var(D_query[i][1], list_source_output, list_target_output)
	for i in range(len(D_support)):
		D_support[i][0] = sentence_replace_var(D_support[i][0], list_source_command, list_target_command)
		D_support[i][1] = sentence_replace_var(D_support[i][1], list_source_output, list_target_output)
	
	D_primitive = list(zip(list_target_command,list_target_output))
	if inc_support_in_query:
		D_query += deepcopy(D_support)		
	return D_support, D_query, D_primitive

def sample_augment_scan_separate(nsupport,nquery,scan_tuples_support_variable,scan_tuples_query_variable,shuffle,nextra=0,inc_support_in_query=False):
	# ** This version takes a SEPARATE set of examples for sampling the support and query items **
	#  Both the query and the support set contain example input/output patterns
	#   Create an episode with shuffled input/output primitives, potentially with augmentation
	#
	# Input
	#  nsupport : number of support items to sample
	#  nquery : number of query items to sample
	#  scan_tuples_support_variable : list of input/output tuples to draw support examples from (in VARIABLE form, as generated by 'load_scan_var')
	#  scan_tuples_query_variable : list of input/output tuples to draw query examples from (in VARIABLE form, as generated by 'load_scan_var')
	#  shuffle : true/false; randomly shuffle the scan input/output primitives, or use semantic alignment?
	#  nextra : number of extra abstract input/output primitives to include
	#  inc_support_in_query : true/false; include the support set in the query set?
	#
	# Output
	#  D_support : list of support input/output pairs; in this case, it's the remapped primitives ONLY
	#  D_query : list of query input/output pairs
	#  D_primitive : list of primitive input/output pairs
	#
	# Sample query items from scan tuples (before variable replacement)
	# Distribute the patterns to query and support items
	scan_tuples_support_variable = deepcopy(scan_tuples_support_variable)
	random.shuffle(scan_tuples_support_variable)
	D_support = scan_tuples_support_variable[:nsupport]

	scan_tuples_query_variable = deepcopy(scan_tuples_query_variable)
	random.shuffle(scan_tuples_query_variable)
	D_query = scan_tuples_query_variable[:nquery]
	
	# Shuffle assignment of primitive commands to primitive actions
	nprim_replace = 4
	scan_primitive_tuples = [('walk','I_WALK'),('look','I_LOOK'),('run','I_RUN'),('jump','I_JUMP')]
	if nextra > 0:
		scan_primitives_extra = [(str(i),'I_'+str(i)) for i in range(1,nextra+1)]
		scan_primitive_tuples += scan_primitives_extra	
	unzip = list(zip(*scan_primitive_tuples))
	list_target_command = list(unzip[0])
	list_target_output = list(unzip[1])
	if shuffle: # shuffle assignment if desired
		random.shuffle(list_target_command)
		random.shuffle(list_target_output)
	list_target_command = list_target_command[:nprim_replace]
	list_target_output = list_target_output[:nprim_replace]

	# Replace placeholders with grounded commands and actions		
	scan_placeholder_tuples = [('primitive'+str(i),'I_ACT_'+str(i)) for i in range(1,nprim_replace+1)]
	list_source_command = [p[0] for p in scan_placeholder_tuples]
	list_source_output = [p[1] for p in scan_placeholder_tuples]
	for i in range(len(D_query)):
		D_query[i][0] = sentence_replace_var(D_query[i][0], list_source_command, list_target_command)
		D_query[i][1] = sentence_replace_var(D_query[i][1], list_source_output, list_target_output)
	for i in range(len(D_support)):
		D_support[i][0] = sentence_replace_var(D_support[i][0], list_source_command, list_target_command)
		D_support[i][1] = sentence_replace_var(D_support[i][1], list_source_output, list_target_output)
	
	D_primitive = list(zip(list_target_command,list_target_output))

	if inc_support_in_query:
		D_query += deepcopy(D_support)		
	return D_support, D_query, D_primitive

if __name__ == "__main__":
	scan_tuples = load_scan_file('simple','train') # load SCAN from file
	scan_tuples_variable = load_scan_var('simple','train') # load SCAN from file
	
	print("Example of primitive replacement...")

	D_support, D_query, D_primitive = sample_augment_scan(5,5,scan_tuples_variable,shuffle=True,nextra=24,inc_support_in_query=False)
	print("New mapping...")
	print(D_primitive)
	print("")
	print("Support")	
	for p in D_support[:4]:
		print(p[0])
		print(p[1])
	print("")
	print("Query")
	for p in D_query[:4]:
		print(p[0])
		print(p[1])