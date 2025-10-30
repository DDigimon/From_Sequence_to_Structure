"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import json
import random
from tqdm import tqdm
import networkx as nx

# import sys
# sys.path.append('../../../../')
from pattern_match import find_pattern_list


# define patterns in topological way
diamond_patterns=['A : D , C : B A , D : B',
                  'B : A D , A : C , C : D']

FFL_patterns=['A : B C , B : C',
              'B : A C , A : C']

nd_diamond_patterns=['A : B C , C : D , D : B',
                    'B : A D , A : C , C : D']

FFL_FBL_patterns=['A : B C D , C : D , D : B',
                'B : D , C : A B D , D : A']

pattern_topos={'FFL':FFL_patterns,'diamond':diamond_patterns,
               'nd-diamond':nd_diamond_patterns,'FFL_FBL':FFL_FBL_patterns}


def parse_directed_graph_to_nx(graph_str):
    G = nx.DiGraph()
    entries = [entry.strip() for entry in graph_str.split(',')]
    for entry in entries:
        if ':' in entry:
            node_str, neighbors_str = entry.split(':')
            node = int(node_str.strip())
            neighbors = [int(n) for n in neighbors_str.strip().split()] if neighbors_str.strip() else []
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        else:
            node = int(entry.strip())
            G.add_node(node)  # In case it's an isolated node with no neighbors
    return G

def ans_generation(triangles,node=5):
    string=''
    for t in triangles:
        if node==3:
            string+=f'{t[0]} {t[1]} {t[2]} , '
        elif node==4:
            string+=f'{t[0]} {t[1]} {t[2]} {t[3]} , '
        elif node==5:
            string+=f'{t[0]} {t[1]} {t[2]} {t[3]} {t[4]} , '
        elif node==6:
            string+=f'{t[0]} {t[1]} {t[2]} {t[3]} {t[4]} {t[5]} , '
    return string[:-1]


method='adj'
define_method_list=['term']
define_method=''
for t in define_method_list:
    define_method+=t
    

scale=100000
sub_scale=50000 # subscale of actual training data

PAD=' <PAD>'
triangle_num=[1]
triangle_num_string=''
for t in triangle_num:
    triangle_num_string+=str(t)

partial=[1]
partial_num=[1]
 
pattern_list=['FFL']
if len(partial)!=len(pattern_list):
    ValueError('partial need be the same length with pattern list')

pattern_count={}
pattern_max_count={}
for idx,p in enumerate(pattern_list):
    for term in define_method_list:
        pattern_count[term+p]=0
        pattern_max_count[term+p]=sub_scale*(partial[idx]/sum(partial))
print(pattern_max_count)

def base_path_chooce():
    # define you own base path here according to the generation settings
    return base_path

overall_txt_all,overall_node_pairs_all,overall_ans_txt_all=[],[],[]
scale_num=int(scale/10000)
if scale_num<1:
    scale_num=1

print(scale_num)
 
overall_valid_txt_all,overall_valid_node_pairs_all,overall_valid_ans_txt_all=[],[],[]

max_ans_length=0

if len(triangle_num)==1:
    ans_max_length=20
else:
    ans_max_length=40
    
if 'cot' in define_method:
    if define_method == 'cot_FFL_FBL':
        ans_max_length=160
    if define_method == 'cot_house':
        ans_max_length=200
    if define_method=='cot_hex2':
        ans_max_length=300

def get_cot(cots,ans,lens_divided=[95,190,200]):
    '''
    Thinking in substructures:
    <P1> description of first substructure <P1_END>
    <P2> description of second substructure <P2_END>
    ...
    <ANS> final answer 
    lens_divided: list of max lengths for each part. 
    Note that the model may sensitive to the length in each part, as we marked in Table 11.

    '''
    txt=''
    max_lens=0
    cot_string='<P1> '
    for idx,c in enumerate(cots):
        # print(idx)
        # print(c)
        for t in c.split(' '):
            if len(cot_string.split(' '))>=lens_divided[idx]-1:
                # max_lens=len(t.split(' '))
                break
            cot_string+=t+' '
        cot_string+=f'<P{idx+1}_END> '
        if len(cot_string.split(' '))<lens_divided[idx]-1:
            for _ in range(lens_divided[idx]-len(cot_string.split(' '))):
                cot_string+='<PAD> '
        cot_string+=f'<P{idx+2}> '
    cot_string+='<ANS> '+ans
    return cot_string
    
    


# start from 3, leaving some cases for testing
for file_i in tqdm(range(3, scale_num+1)):
        for terms in define_method_list:
            for p in pattern_list:
                    base_path=base_path_chooce(pattern_list,triangle_num,p)
                    with open(os.path.join(base_path,f'tiny_{file_i}_idx.json'),'r') as f:
                        graph_dicts=json.load(f)

                    with open(os.path.join(base_path,f'tiny_{file_i}_graphs_description_{method}.pkl'),'rb') as f:
                        graph_description_list=pickle.load(f)
                        
                    with open(os.path.join(base_path,f'tiny_{file_i}_ans.pkl'),'rb') as f:
                        ans_list=pickle.load(f)
                    
                    for t in triangle_num:
                        if p + str(t) in graph_dicts:
                            idx=graph_dicts[p+str(t)]
                            if pattern_count[terms+p]>pattern_max_count[terms+p]:continue
                            pattern_count[terms+p]+=len(idx)
                            overall_txt_all.extend([graph_description_list[i] for i in idx])
                            if terms=='exterm':
                                overall_ans_txt_all.extend(['<START_T> '+p+' '+ans_list[i][p] for i in idx])
                            
                            elif terms=='cot_FFL_FBL':
                                for i in idx:
                                    graphs = graph_description_list[i]
                                    G = parse_directed_graph_to_nx(graphs)
                                    FFL_triangle = find_pattern_list(G,'FFL')
                                    FFL_triangle = ans_generation(FFL_triangle,3)
                                    FBL_triangle = find_pattern_list(G,'FBL')
                                    FBL_triangle = ans_generation(FBL_triangle,3)
                                    if p == 'FFL_FBL':
                                        txt = get_cot([FFL_triangle,FBL_triangle],ans_list[i][p],lens_divided=[55,150])
                                    elif p=='cross2':
                                        txt = get_cot([FFL_triangle,FBL_triangle],ans_list[i][p],lens_divided=[95,150])
                                    if len(txt.split(' '))>max_ans_length:
                                        max_ans_length=len(txt.split(' '))  
                                    overall_ans_txt_all.append(txt)
                            elif terms=='cot_hex2':
                                for i in idx:
                                    graphs = graph_description_list[i]
                                    G = parse_directed_graph_to_nx(graphs)
                                    FFL_triangle = find_pattern_list(G,'d-tr2')
                                    FFL_triangle2 = find_pattern_list(G,'d-tr')
                                    square = find_pattern_list(G,'d-sq')
                                    ans = find_pattern_list(G,'hex2')

                                    FFL_triangle = ans_generation(FFL_triangle,3)
                                    FFL_triangle2 = ans_generation(FFL_triangle2,3)
                                    square = ans_generation(square,4)
                                    ans = ans_generation(ans,6)
                                    txt = get_cot([FFL_triangle,FFL_triangle2,square],ans_list[i][p],lens_divided=[80,180,290])
                                    if len(txt.split(' '))>max_ans_length:
                                        max_ans_length=len(txt.split(' '))  
                                    overall_ans_txt_all.append(txt)  
                            elif terms=='cot_house':
                                for i in idx:
                                    graphs = graph_description_list[i]
                                    G = parse_directed_graph_to_nx(graphs)
                                    FFL_triangle = find_pattern_list(G,'d-tr')
                                    FFL_triangle = ans_generation(FFL_triangle,3)
                                    FBL_triangle = find_pattern_list(G,'d-sq')
                                    FBL_triangle = ans_generation(FBL_triangle,4)

                                    txt = get_cot([FFL_triangle,FBL_triangle],ans_list[i][p],lens_divided=[75,190])

                                    if len(txt.split(' '))>max_ans_length:
                                        max_ans_length=len(txt.split(' '))  
                                    overall_ans_txt_all.append(txt)
                            
                            else:
                                overall_ans_txt_all.extend([ans_list[i][p] for i in idx])
                            
                            # some experiment code for what kinds of questions are asked to transformers.
                            if terms=='s':
                                overall_node_pairs_all.extend([ans_list[i][p].split(' ')[0] for i in idx])
                            elif terms=='e':
                                overall_node_pairs_all.extend([ans_list[i][p].split(' ')[-2] for i in idx])
                            elif terms=='se':
                                overall_node_pairs_all.extend([f"{ans_list[i][p].split(' ')[0]} {ans_list[i][p].split(' ')[-2]}" for i in idx])
                            elif terms=='topo' or terms=='exterm':
                                overall_node_pairs_all.extend([random.choice(pattern_topos[p]) for _ in range(len(idx))])
                            elif terms=='topo0':
                                overall_node_pairs_all.extend([pattern_topos[p][0] for _ in range(len(idx))])
                            elif terms=='topo1':
                                overall_node_pairs_all.extend([pattern_topos[p][1] for _ in range(len(idx))])
                            else:
                                overall_node_pairs_all.extend([p for _ in range(len(idx))])
                            if len(overall_node_pairs_all)>sub_scale:break
                        else:
                            print(f'{p,str(t)} not in dictions')
                    if len(overall_node_pairs_all)>sub_scale:break
            if len(overall_node_pairs_all)>sub_scale:break

print(f'num of training {len(overall_txt_all)}, num of ans {len(overall_ans_txt_all)}, num of node pairs {len(overall_node_pairs_all)}')

for key in graph_dicts.keys():
    print(key,len(graph_dicts[key]))

if method == 'edge':
    question_max_length = 1200
elif method == 'adj':
    question_max_length = 650
if 'tiny' in base_path:
    if method == 'edge':
        question_max_length = 400
    elif method == 'adj':
        question_max_length =200
if 'mid' in base_path:
    if method == 'edge':
        question_max_length = 1000
    elif method == 'adj':
        question_max_length =500
if 'large' in base_path:
    if method == 'edge':
        question_max_length = 2000
    elif method == 'adj':
        question_max_length =1000
if 'all' in base_path:
    if method == 'edge':
        question_max_length = 2000
    elif method == 'adj':
        question_max_length =1000
if 'baby' in base_path:
    if method == 'edge':
        question_max_length = 100
    elif method == 'adj':
        question_max_length =100
        
# question_max_length=250
START_Q='<START_Q> '
END_Q=' <END_Q>'
START_A='<START_A> '
def construct_Q(txts,pairs):
    questions=''
    for idx,t in tqdm(enumerate(txts),total=len(txts)):
        pad_string=''
        for i in range(question_max_length-(len(t.split(' '))+len(pairs[idx].split(' ')))-3):
            pad_string+=PAD
        single_data=START_Q+t+END_Q+pad_string+' '+pairs[idx]+' '+START_A
        
        questions+=single_data
    return questions[:-1]


END=' <END>'

import random
import string
def construct_A(ans):
    ans_data=''
    for idx,a in tqdm(enumerate(ans),total=len(ans)):
        
        if ans_max_length-len(a.split(' '))-1<0:
            tmp_data=a.split(' ')[:ans_max_length]
            s=''
            for t in tmp_data:
                s+=t+' '
            single_data=s
        else:
            pad_string=''
            for i in range(ans_max_length-len(a.split(' '))-1):
                pad_string+=PAD
            single_data=a+ END+pad_string+' '
        ans_data+=single_data
    print(ans_data.split(' ')[-ans_max_length-2:])
    print(single_data.split(' '))
    return ans_data[:-1]

val=int(sub_scale*0.1)
overall_valid_txt_all= overall_txt_all[:val]
overall_valid_node_pairs_all=overall_node_pairs_all[:val]
overall_valid_ans_txt_all=overall_ans_txt_all[:val]

overall_txt_all= overall_txt_all[val:]
overall_node_pairs_all=overall_node_pairs_all[val:]
overall_ans_txt_all=overall_ans_txt_all[val:]

step_question=construct_Q(overall_txt_all,overall_node_pairs_all)
step_ans=construct_A(overall_ans_txt_all)

print(f'num of training {len(overall_txt_all)}, num of validation {len(overall_valid_txt_all)}')

valid_step_question=construct_Q(overall_valid_txt_all,overall_valid_node_pairs_all)
valid_step_ans=construct_A(overall_valid_ans_txt_all)
data=step_question+' '+step_ans+' '+valid_step_question+' '+valid_step_ans
chars = sorted(list(set(data.split(' '))))
vocab_size = len(chars)
print("all the unique characters:", ' '.join(chars))
print(f"vocab size: {vocab_size:,}")
# exit()
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    s=s.split(' ')
    # print(len(s))
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ' '.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


# encode both to integers
train_question_ids=encode(str(step_question))

train_ans_ids=encode(str(step_ans))

valid_question_ids=encode(str(valid_step_question))
valid_ans_ids=encode(str(valid_step_ans))

print(f"train has {len(train_question_ids):,} tokens;"+f"ans {len(train_ans_ids):,} tokens")
print(f"val has {len(valid_question_ids):,} tokens;"+f"ans {len(valid_ans_ids):,} tokens")

print(pattern_count)


pattern_name_list=''
for p in pattern_list:
    pattern_name_list+=p

train_question_ids = np.array(train_question_ids, dtype=np.uint16)
valid_question_ids = np.array(valid_question_ids, dtype=np.uint16)
print(valid_question_ids)
train_ans_ids = np.array(train_ans_ids, dtype=np.uint16)
valid_ans_ids = np.array(valid_ans_ids, dtype=np.uint16)
# save to bin files
train_question_ids.tofile(os.path.join(os.path.dirname(__file__), f'train_question_{define_method}_{pattern_name_list}_{triangle_num_string}_{method}_{sub_scale}.bin'))
valid_question_ids.tofile(os.path.join(os.path.dirname(__file__), f'val_question_{define_method}_{pattern_name_list}_{triangle_num_string}_{method}.bin'))
train_ans_ids.tofile(os.path.join(os.path.dirname(__file__), f'train_ans_{define_method}_{pattern_name_list}_{triangle_num_string}_{method}_{sub_scale}.bin'))
valid_ans_ids.tofile(os.path.join(os.path.dirname(__file__), f'val_ans_{define_method}_{pattern_name_list}_{triangle_num_string}_{method}.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'max_questions':question_max_length,
    'max_ans':ans_max_length
}
with open(os.path.join(os.path.dirname(__file__), f'my_meta_{define_method}_{pattern_name_list}_{triangle_num_string}_{method}.pkl'), 'wb') as f:
    pickle.dump(meta, f)

