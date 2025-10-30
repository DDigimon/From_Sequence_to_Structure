import networkx as nx
import numpy as np

from pattern_match import find_pattern_list
import os
import pickle
import json
import random
import os



def generate_random_graph_fixed_edges(n, num_edges,directed=False):
    max_edges = n * (n - 1) // 2  
    if num_edges > max_edges:
        raise ValueError(f"Too many edges: max possible edges for {n} nodes is {max_edges}")

    if directed:
        dag=nx.DiGraph()
        dag.add_nodes_from(range(n))
        while dag.number_of_edges() < num_edges:
            u, v = random.sample(range(n), 2)
            
            if not dag.has_edge(u, v): 
                dag.add_edge(u, v)
        return dag
    else:
        G = nx.Graph()
    G.add_nodes_from(range(n))
    possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edges = random.sample(possible_edges, num_edges)
    G.add_edges_from(edges)
    
    return G


def generate_adj_list(g):
    '''
    Adjacency list format:
    node1 : neighbor1 neighbor2 neighbor3 ...
    node2 : neighbor1 neighbor2 ...
    '''
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
    
    return txt[1:]

def generate_edge_list(g):
    '''
    Edge list format:
    (node1, node2) (node3, node4) ...'''
    txt=str(g.edges())[1:-1].replace(', ',' ')
    txt=txt.replace(') (',' | ')
    txt=txt.replace('(','')
    txt=txt.replace(')','')
    return txt


def ans_generation(triangles,node=5):
    '''
    Not a good code formulation, but can run
    '''
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

graph_dicts={}
graph_list=[]
graph_description_adj_list=[]
graph_description_edge_list=[]
ans_list=[]
graph_set=set()
max_num=50000000
sub_count=0
counts=0
counts_bag=set()
max_length=0
max_length_edge=0
max_q_length=0
max_q_length_e=0
max_a_length=0
name='FFL' # FFL is triangle. More patterns check pattern_match.py
base_path=f'tiny_{name}'
di=True

if os.path.exists(base_path)==False:
    os.makedirs(base_path)

while True:

    n=random.randint(3,16) # can change the graph size range here
    if di:
        max_edges = int((n * (n - 1) // 2))
    else:
        max_edges = int((n * (n - 1) // 2)*0.5)
    edges_num=random.randint(2, max_edges)
    graph=generate_random_graph_fixed_edges(n,edges_num,directed=di)
    description_adj=generate_adj_list(graph)
    description_edge=generate_edge_list(graph)
    if description_adj not in graph_set:
        graph_set.add(description_adj)
        target_pattern=find_pattern_list(graph,name)
        if len(target_pattern)==0:continue
        if len(target_pattern)>20:continue # exclude too many pattern cases
        
        if name+str(len(target_pattern)) not in graph_dicts:
            graph_dicts[name+str(len(target_pattern))]=[]
        graph_dicts[name+str(len(target_pattern))].append(sub_count)
        diamond_ans=ans_generation(target_pattern,3)


        ans_list.append({name:diamond_ans})
        graph_description_adj_list.append(description_adj)
        graph_description_edge_list.append(description_edge)
        description_length_adj=len(description_adj.split(' '))
        description_length_edge=len(description_edge.split(' '))
        ans_length=len(diamond_ans.split(' '))
        
        max_sentence_length_adj=description_length_adj+ans_length
        if max_sentence_length_adj>max_length:
            max_length=max_sentence_length_adj
        max_sentence_length_edge=description_length_edge+ans_length
        if max_sentence_length_edge>max_length_edge:
            max_length_edge=max_sentence_length_edge
            
        if description_length_adj>max_q_length:
            max_q_length=description_length_adj
        if description_length_edge>max_q_length_e:
            max_q_length_e=description_length_edge
        if ans_length>max_a_length:
            max_a_length=ans_length
        counts+=1
        sub_count+=1
        
    if counts%10000==0 and counts!=0 and counts not in counts_bag:
        '''
        Pacakaging the dataset every 10000 samples
        1. graphs.pkl : list of graphs in adjacency list format
        2. idx.json : dictionary of pattern name + number of patterns : list of indices
        3. ans.pkl : list of answers corresponding to each graph
        4. graphs_description_adj.pkl : list of graph descriptions in adjacency list format
        5. graphs_description_edge.pkl : list of graph descriptions in edge list format
        '''
        print(counts)
        with open(os.path.join(base_path,f'tiny_{int(counts/10000)}_graphs.pkl'),'wb') as f:
            pickle.dump(graph_list,f)
        with open(os.path.join(base_path,f'tiny_{int(counts/10000)}_idx.json'),'w') as f:
            json.dump(graph_dicts,f)

        with open(os.path.join(base_path,f'tiny_{int(counts/10000)}_graphs_description_adj.pkl'),'wb') as f:
            pickle.dump(graph_description_adj_list,f)
            
        with open(os.path.join(base_path,f'tiny_{int(counts/10000)}_ans.pkl'),'wb') as f:
            pickle.dump(ans_list,f)
        counts_bag.add(counts)
        
        for key in graph_dicts.keys():
            print(key,len(graph_dicts[key]))
        graph_dicts={}
        graph_list=[]
        graph_description_adj_list=[]
        graph_description_edge_list=[]
        ans_list=[]
        sub_count=0
        print(f'saved, max length:{max_length_edge}, {max_q_length}, {max_q_length_e}, {max_a_length},')
    if counts==max_num:
        break
print('finished')

