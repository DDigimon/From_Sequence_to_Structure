import os
import torch
from model import GPTConfig, GPT
import pickle
from tqdm import tqdm
import networkx as nx
import json
import sys
sys.path.append('../../')
from pattern_match import find_pattern_list
from copy import deepcopy

START_Q='<START_Q> '
END_Q=' <END_Q>'
START_A='<START_A>'
PAD=' <PAD>'
IN_LABEL='T'
OTHERS='O'
TARGET='A'
PRED_ANS='P'
PAIRS='N'
COMS='S'

diamond_patterns=['A : D , C : B A , D : B',
                  'B : A D , A : C , C : D',
                  ': , : , : ',
                  'C',
                  ': , : , : D']

FFL_patterns=['A : B C , B : C',
              'B : A C , A : C',
              ': , : ',
              'C',
              ': , : C']


nd_diamond_patterns=['A : B C , C : D , D : B',
                    'B : A D , A : C , C : D',
                    '<PAD> : <PAD> <PAD> , <PAD> : <PAD> , <PAD> : <PAD>',
                    'A']

FFL_FBL_patterns=['A : B C D , C : D , D : B',
                'B : D , C : A B D , D : A',
                '<PAD> : <PAD> , <PAD> : <PAD> <PAD> <PAD> , <PAD> : <PAD>',
                'A']

pattern_topos={'FFL':FFL_patterns,'diamond':diamond_patterns,
               'nd-diamond':nd_diamond_patterns,'FFL_FBL':FFL_FBL_patterns}


def load_basic_infor(checkpoint,meta_path):
    
    # checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

        
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    return model,meta


def construct_Q(txts,pairs,ans_nodes):
    questions=''
    pad_string=''
    nodes=pairs.split(' ')[2:]
    labels=''
    ans_nodes=ans_nodes.split(' ')
    for idx,txt in enumerate(txts.split('|')):
        nums=2
        if txt in ans_nodes:
            labels+=IN_LABEL
        else:
            labels+=IN_LABEL
        

        if idx!=len(txts.split('|'))-1:
            labels+=COMS
    
    for i in range(question_max_length-(len(txts.split(' '))+len(pairs.split(' ')))-2):
        pad_string+=PAD
    single_data=START_Q+txts+pad_string+' '+pairs+' '+START_A
    pad_string=''
    for i in range(ans_max_length):
        pad_string+=PAD
    questions+=single_data+pad_string

    return questions, labels

def generate_adj_list(g):
    txt=''
    adjacency_list = list(nx.generate_adjlist(g))
    for line in adjacency_list:
        line=line.split(' ')
        strat_node=line[0]
        txt+=', '+strat_node+' : '
        for idx in range(len(line)):
            if line[idx]==' ':continue
            if idx==0:continue
            txt+=line[idx]+' '
        # print(line)
    return txt

def generate_edge_list(g):

    txt=str(g.edges())[1:-1].replace(', ',' ')
    
    txt=txt.replace(') (',' | ')
    txt=txt.replace('(','')
    txt=txt.replace(')','')
    return txt


def base_path_chooce(pattern_list,triangle_num,pattern,form='small'):
    '''
    set your own path here
    '''
    return base_path


def load_data(triangle_num, pattern_list, method,terms_define='se',trigger_construct=False,dens_eval=False,form='small',test_num=0):
    x_list=[]
    x_length=[]
    ans=[]
    pattern_mark=[]
    # for idx,g in enumerate(graph_list):
    graph_txt_dicts={}
    vect_dicts={}
    idx_mark=[]
    
    if form == 'tiny':
        ends = 3
    else:
        ends = 30
    for file_idx in range(1,ends):

        for p in pattern_list:
            base_path=base_path_chooce(pattern_list,triangle_num,p,form=form)
            with open(os.path.join(base_path,f'tiny_{file_idx}_idx.json'),'r') as f:
                graph_dicts=json.load(f)

            with open(os.path.join(base_path,f'tiny_{file_idx}_graphs_description_{method}.pkl'),'rb') as f:
                graph_description_list=pickle.load(f)
                
            with open(os.path.join(base_path,f'tiny_{file_idx}_ans.pkl'),'rb') as f:
                ans_list=pickle.load(f)
                
            for terms in defined_method_list:
                if terms=='term' and len(defined_method_list)==2:continue
                if p+terms not in graph_txt_dicts:
                    graph_txt_dicts[p+terms]=[]
                for t in triangle_num:
                    if p+terms not in vect_dicts:
                        vect_dicts[p+terms]=[]
                    idx=graph_dicts[p+str(t)]
                    descriptions=[graph_description_list[i] for i in idx]
                    graph_txt_dicts[p+terms]=descriptions

        for p in pattern_list:
            base_path=base_path_chooce(pattern_list,triangle_num,p,form=form)
            with open(os.path.join(base_path,f'tiny_{file_idx}_idx.json'),'r') as f:
                graph_dicts=json.load(f)

            with open(os.path.join(base_path,f'tiny_{file_idx}_graphs_description_{method}.pkl'),'rb') as f:
                graph_description_list=pickle.load(f)
                
            with open(os.path.join(base_path,f'tiny_{file_idx}_ans.pkl'),'rb') as f:
                ans_list=pickle.load(f)
            for t in triangle_num:
                idx=graph_dicts[p+str(t)]# [:1500]
            
                ans.extend([ans_list[i][p] for i in idx])
                current_ans=[ans_list[i][p] for i in idx]
                descriptions=[graph_description_list[i] for i in idx]
                pattern_mark.extend([p for _ in idx])
                orther_patterns=[patt for patt in pattern_list if patt!=p]
                for i,txt in enumerate(descriptions):
                    flag=True
                    for patt in orther_patterns:
                        if txt not in graph_txt_dicts[patt+terms]:
                            flag=False
                    if flag==True:
                        idx_mark.append(len(x_list))
                    if terms_define=='se':
                        nodes_str=f"{ans[i].split(' ')[0]} {ans[i].split(' ')[-2]}"
                    elif terms_define=='topo':
                        nodes_str=deepcopy(pattern_topos[p][3])
                    elif terms_define=='topo0':
                        nodes_str=pattern_topos[p][0]
                    elif terms_define=='topo1':
                        nodes_str=pattern_topos[p][2]
                    else:nodes_str=p
                    start, labels=construct_Q(txt,nodes_str,current_ans[i])
                    x_length.append(len(txt.split(' ')))
                    start_ids = encode(start.split(' '))
                    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
                    x_list.append(x)
    if test_num==0:
        if form == 'small':
            data_test_num=30000
        else:
            data_test_num=1000
    else:
        data_test_num=test_num
        
    ans=ans[:data_test_num]
    x_list=x_list[:data_test_num]
    return x_length,x_list,ans,pattern_mark,idx_mark#,nodes_list


def node_search(strings,nodes=4):
    start_idx=strings.index('<START_A>')
    idx_list=[]
    if '<END>' not in strings:
        if nodes==4:
            idx_list.append(['0','0','0','0'])
        elif nodes==3:
            idx_list.append(['0','0','0'])
    else:
        end_idx=strings.index('<END>')
        answers=strings[start_idx + 1:end_idx]
        for a in answers:
            idx_list.append(a)
    return idx_list
    
def verify_specific_path(G, path, nodes):
    is_path = len(path) > 1 and all(G.has_edge(path[i], path[i+1]) for i in range(len(path) - 1))
    is_target = len(path) > 1 and ((path[0] == nodes[0] and path[-1] == nodes[1]) or (path[0] == nodes[1] and path[-1] == nodes[0]))
    return is_path and is_target

def get_ans(txt,node_num=0):
    # print(txt)
    ans=[]
    txt=txt.split(' ')
    if '<START_A>' in txt:
        txt=txt[txt.index('<START_A>')+1:]
    if '<ANS>' in txt:
        last_index = len(txt) - 1 - txt[::-1].index('<ANS>')
        indices = [i for i, x in enumerate(txt) if x == '<ANS>']
        for i in indices:
            if txt[i+node_num+2] == '<END>' or i+node_num==len(txt):
                last_index = i
                break

        txt=txt[last_index+1:]
    if '<END>' in txt:
        txt=txt[:txt.index('<END>')]
        tmp_txt=[]
        txt.append(',')
        for t in txt:
            if ',' in t and t!=',':
                t=t.split(',')[0]
                
            if t==',':
                if len(tmp_txt)!=0:
                    ans.append(tmp_txt)
                    tmp_txt=[]
            else:
                tmp_txt.append(t) 
        return ans
    else:
        return ans
    
def get_ground_turth(txt):
    ans=[]
    txt=txt.split(',')
    for t in txt:
        if len(t)==0:continue
        t=t.split(' ')# .remove('')
        ans.append([item for item in t if item != ''])
    return ans


def precision(pred,ans):
    score=[]
    for triple_idx,triple in enumerate(ans):
        if triple in pred:
            score.append(1)
        else:
            score.append(0)
    if len(score)==0:
        return 0
    return sum(score)/len(score)



def verify_ans(pred,ground_truth,given_pattern_mark=None):
    
    acc_dicts={}
    ground_truth=get_ground_turth(ground_truth)
    node_num=len(ground_truth[0])
    pred_ans=get_ans(pred,node_num=node_num)
    acc=0
    in_count={}
    print(pred_ans)
    print(ground_truth)
    if len(pred_ans)==0:
        print(pred)
    for triple_idx,triple in enumerate(ground_truth):
        if triple_idx not in acc_dicts:
            acc_dicts[triple_idx]=[]
        if triple_idx in pred_ans:
            if pred_ans[triple_idx]==ground_truth[triple_idx]:
                acc_dicts[triple_idx].append(1)
            else:
                acc_dicts[triple_idx].append(0)
    p=precision(pred_ans,ground_truth)
    r=precision(ground_truth,pred_ans)
    if p==0 and r==0:
        f1=0
    else:
        f1=2*p*r/(p+r)
    if len(ground_truth) not in in_count:
        in_count[len(ground_truth)]=[]
    in_count[len(ground_truth)].append(f1)
    if 's'+str(len(ground_truth[0])) not in in_count:
        in_count['s'+str(len(ground_truth[0]))]=[]
    in_count['s'+str(len(ground_truth[0]))].append(f1)
    if given_pattern_mark is not None:
        if given_pattern_mark not in in_count:
            in_count[given_pattern_mark]=[]
        in_count[given_pattern_mark].append(f1)
    return acc_dicts,f1,in_count

    

def evaluating(model,x_length,sentence_id,ans,pattern_mark,idx_mark):
    
    acc_list=[]
    overall_acc_dicts={}
    incount_dicts={}
    idx_score=[]
    for idx,g in tqdm(enumerate(ans),total=len(ans)):
        if x_length[idx]>question_max_length: 
            continue
        y = model.generate(sentence_id[idx], ans_max_length,max_question_length=question_max_length)

        response=decode(y[0].tolist())# .split(' ')[question_max_length:]
        acc_dicts,acc,incount=verify_ans(response,ans[idx],given_pattern_mark=pattern_mark[idx])
        for key in incount.keys():
            if key not in incount_dicts:
                incount_dicts[key]=[]
                
            incount_dicts[key].extend(incount[key])
        acc_list.append(acc)
        if idx in idx_mark:
            idx_score.append(acc)
        for key in acc_dicts.keys():
            if key not in overall_acc_dicts:
                overall_acc_dicts[key]=[]
            overall_acc_dicts[key].extend(acc_dicts[key])
        # break
    acc_score=sum(acc_list)/len(acc_list)
    # idx_score=sum(idx_score)/len(idx_score)
    for key in overall_acc_dicts.keys():
        if len(overall_acc_dicts[key])!=0:
            overall_acc_dicts[key]=sum(overall_acc_dicts[key])/len(overall_acc_dicts[key])
        else:
            overall_acc_dicts[key]=0

    for key in incount_dicts.keys():

        incount_dicts[key]=sum(incount_dicts[key])/len(incount_dicts[key])
        print(key,incount_dicts[key])
        
    return acc_score,incount_dicts,idx_score
    print(acc_score,len(acc_list))

if __name__ == "__main__":
    method='adj'
    out_dir = 'out-shakespeare-char'
    device='cuda:0'
    triangle_num, pattern_list, method=[1],['poly'],'edge'
    defined_method_list=['term']
    define_method=''
    epoch = 0
    for d in defined_method_list:
        define_method+=d
    seed=2
    
    pattern_name_list=''
    for p in pattern_list:
        pattern_name_list+=p
    triangle_num_string=''
    for t in triangle_num:
        triangle_num_string+=str(t)
    # scale=2000000
    sub_scale=350000
    eval_data=int(sub_scale/20)
    n_embd=192
    n_layer=5
    n_head=12
    test_num=0
    results_eval='re'
    model_scale={384:'small',768:'mid',1024:'large',192:'tiny'}
    base_path=f'/egr/research-dselab/shared/daixinna/nano-pattern/tiny_models'
    out_dir=base_path+f'/model_{pattern_name_list}_{triangle_num_string}_{method}_{define_method}_{model_scale[n_embd]}_{n_layer}_{n_head}_{sub_scale}_{seed}'
    if epoch!=0:
        ckpt_path = os.path.join(out_dir, f'{epoch}_model.pt')
    else:
        ckpt_path = os.path.join(out_dir, f'model.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    meta_path = os.path.join('data', checkpoint['config']['dataset'], f'{base_path}/my_meta_{define_method}_{pattern_name_list}_{triangle_num_string}_{method}.pkl')

    model,meta=load_basic_infor(checkpoint,meta_path)
    model.to(device)
    model.eval()
    print(meta['stoi'])

    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ' '.join([itos[i] for i in l])

    question_max_length=meta['max_questions']
    ans_max_length=meta['max_ans']
    print('max question length:',question_max_length)
    print('max ans length:',ans_max_length)
    if 'cot' in define_method:
        ans_max_length=meta['max_ans']+100
    acc_dicts_sum={}
    acc_score_list=[]
    idx_scores=[]
    form=model_scale[n_embd]
    if results_eval=='re':
        for t in defined_method_list:
            if len(defined_method_list)==2 and t=='term':continue
            x_length,sentence_id,ans,pattern_mark,idx_mark=load_data(triangle_num, pattern_list, method,terms_define=t,form=form,test_num=test_num)
            acc_score,acc_dicts,idx_score=evaluating(model,x_length,sentence_id,ans,pattern_mark,idx_mark)
            acc_score_list.append(acc_score)
            # idx_scores.append(idx_score)
            acc_dicts_sum[t]={}
            for key in acc_dicts.keys():
                if key not in acc_dicts_sum:
                    acc_dicts_sum[t][key]=[]
                acc_dicts_sum[t][key].append(acc_dicts[key])

        print(acc_score)
        print(acc_dicts)
        # print(idx_scores)
        
        print(sum(acc_score_list)/len(acc_score_list))
        for key in acc_dicts_sum.keys():
            for key2 in acc_dicts_sum[key].keys():
                print(key,key2,sum(acc_dicts_sum[key][key2])/len(acc_dicts_sum[key][key2]))
    else:
        for t in defined_method_list:
            x_length,sentence_id,ans,pattern_mark,idx_mark=load_data(triangle_num, pattern_list, method,terms_define=t)
  

    print(out_dir)
    print(epoch)
    
