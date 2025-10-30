import networkx as nx
from networkx.algorithms import isomorphism
from graph_nlp import graph_txt

pattern_description={'triangle':'triangle (A motif consisting of three nodes where each node is connected to the other two, forming a triangle)',
                     'diamond':'diamond (A four-node motif with five edges)',
                     'tailed-triangle':'tailed triangle (A triangle with an additional node connected to one of the vertices of the triangle)',
                     'square':'square (A 4-node cycle where each node is connected to exactly two other nodes)',
                     'house':'house (A motif resembling the shape of a house with 5 nodes and 6 edges. The vertices and edges are arranged such that there is a triangular "roof" on top of a square or rectangular "base."',
                     'FFL':'3-node Feed-Forward Loop (A three-node directed motif in which one source node influences a target node through two distinct pathways)',
                     'FBL':'3-node Feedback loop (A directed cycle where the nodes form a loop)',
                     'vs':'V-structure (Two nodes have directed edges pointing toward a common target node)',
                     'd-diamond':'direceted diamond (A 4-node motif in a directed graph where one node has directed edges to two intermediate nodes, and both of those intermediate nodes have directed edges to a common target node.)',
                     }

pattern_output_form={'diamond':'The detected patterns are: [(#1, #2, #3, #4), ...]',
            'square':'The detected patterns are: [(#1, #2, #3, #4), ...]',
            'tailed-triangle':'The detected patterns are: [(#1, #2, #3, #4), ...]',
            'house':'The detected patterns are: [(#1, #2, #3, #4, #5), ...]',
            'triangle':'The detected patterns are: [(#1, #2, #3), ...]',
            'FFL':'The detected patterns are: [(#1, #2, #3), ...]',
            'FBL':'The detected patterns are: [(#1, #2, #3), ...]',
            'vs':'The detected patterns are: [(#1, #2, #3), ...]',
            'd-diamond':'The detected patterns are: [(#1, #2, #3, #4), ...]',
            'diamond_a':'The detected patterns are: [(#1, #2, #3, #4, #5), ...]'}


def direct_judge(patterns):
    if 'FFL' in patterns or 'FBL' in patterns or 'd-diamond' in patterns or 'vs' in patterns:
        direction=True
    else:
        direction=False
    return direction

def _triangle():
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
    return G

def _triangle_tailed():
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'),('D','C')])
    return G

def _square():
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'),('D','A')])
    return G

def _house():
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'),('D','A'),('C','E'),('E','D')])
    return G

def _hexagon():
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'),('D','E'),('E','F'),('F','A')])
    return G

def _diamond():
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'D'),('B', 'D')])
    return G


def _diamond_a():
    G = nx.Graph()
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'D'),('B', 'D'),('D','E')])
    return G

def _FFL():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C'])
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C')])
    return G

def _tFFL():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D'])
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'),('A','D')])
    return G
def _tFFL_vs():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D'])
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'),('D','A')])
    return G

def _FBL():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C'])
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
    return G


def _vs():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C'])
    G.add_edges_from([('A', 'B'), ('C', 'B')])
    return G

def _ddamond():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D'])
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'D'),('D','C')])
    return G

def _nddamond():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D'])
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'D'),('C','D')])
    return G

def _tree():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D','E'])
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'),('B','E')])
    return G

def _FFL_FBL():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D'])
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'D'),('C','D'),('D','B')])
    return G

def _cross():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D'])
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'D'),('C','D'),('D','B'),('A','C')])
    return G

def _cross2():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D'])
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'D'),('C','D'),('D','B'),('C','A')])
    return G

def _complex():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D','E','F','G','H','I','J'])
    G.add_edges_from([('A', 'B'), ('A','C'),('B','C'),('C','D'),('D','E'),('E','F'),('F','G'),('D','G'),('G','H'),('H','I'),('H','J'),('I','J'),('J','B')])
    return G

def _path():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D'])
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
    return G

def _path5():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D','E'])
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'),('D','E')])
    return G

def _path_inv():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D'])
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('D', 'C')])
    return G


def _d_house():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D','E'])
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'E'),('C','D'),('D','E'),('B','D')])
    return G

def _d_sq():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B','C','D'])
    G.add_edges_from([('A', 'B'), ('A', 'D'),('C','D'),('B','C')])
    return G

def _d_tr():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B','C'])
    G.add_edges_from([('A', 'B'), ('B','C'),('A','C')])
    return G

def _d_tr2():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B','C'])
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'),])
    return G

def _poly():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D','E'])
    G.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'E'),('C','D'),('D','E')])
    return G


def _hex():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D','E','F'])
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'C'), ('B', 'E'), ('D', 'E'),('D','F'),('F','E')])
    return G

def _hex2():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D','E','F'])
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D'), ('D', 'B'), ('C', 'F'),('D','E'),('E','F')])
    return G

def _pet():
    G = nx.DiGraph()
    G.add_nodes_from(['A', 'B', 'C','D','E'])
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'C'), ('B', 'E'), ('D', 'E')])
    return G

def pattern_generation(name):
    if name=='triangle':
        return _triangle()
    if name=='diamond':
        return _diamond()
    if name=='diamond_a':
        return _diamond_a()
    if name=='FFL':
        return _FFL()
    if name=='tFFL':
        return _tFFL()
    if name=='tFFL_vs':
        return _tFFL_vs()
    if name=='path':
        return _path()
    if name=='path5':
        return _path5()
    if name=='path_inv':
        return _path_inv()
    if name=='FFL_FBL':
        return _FFL_FBL()
    if name=='tailed-triangle':
        return _triangle_tailed()
    if name=='square':
        return _square()
    if name=='house':
        return _house()
    if name=='FBL':
        return _FBL()
    if name=='vs':
        return _vs()
    if name=='d-diamond':
        return _ddamond()
    if name=='d-house':
        return _d_house()
    if name=='d-sq':
        return _d_sq()
    if name=='d-tr':
        return _d_tr()
    if name=='d-tr2':
        return _d_tr2()
    if name=='nd-diamond':
        return _nddamond()
    if name=='poly':
        return _poly()
    if name=='hexagon':
        return _hexagon()
    if name=='hex':
        return _hex()
    if name=='hex2':
        return _hex2()
    if name=='pet':
        return _pet()
    if name=='cross':
        return _cross()
    if name=='cross2':
        return _cross2()
    if name=='tree':
        return _tree()
    

def find_pattern_list(target_graph,pattern_name):
    if 'claim' in pattern_name:
        pattern_name=pattern_name.split('_')[1]
    directed=direct_judge(pattern_name)
    pattern_graph=pattern_generation(pattern_name)
    if directed==False:
        GM = isomorphism.GraphMatcher(target_graph, pattern_graph)
    else:
        GM = isomorphism.DiGraphMatcher(target_graph, pattern_graph,node_match=None,edge_match=None)
    matches = list(GM.subgraph_isomorphisms_iter())
    triangles_networkx=[]
    for m in matches:
        pattern=tuple(m.keys())
        
        triangles_networkx.append(pattern)
    if len(triangles_networkx)!=0:
        if len(triangles_networkx[0])==3:
            triangles_networkx = sorted(triangles_networkx, key=lambda triangles_networkx: (triangles_networkx[0], triangles_networkx[1]))
        elif len(triangles_networkx[0])==4:
            triangles_networkx = sorted(triangles_networkx, key=lambda triangles_networkx: (triangles_networkx[0], triangles_networkx[1], triangles_networkx[2]))
        elif len(triangles_networkx[0])==5:
            triangles_networkx = sorted(triangles_networkx, key=lambda triangles_networkx: (triangles_networkx[0], triangles_networkx[1], triangles_networkx[2], triangles_networkx[3]))
        elif len(triangles_networkx[0])==6:
            triangles_networkx = sorted(triangles_networkx, key=lambda triangles_networkx: (triangles_networkx[0], triangles_networkx[1], triangles_networkx[2], triangles_networkx[3], triangles_networkx[4]))

    return triangles_networkx

if __name__=='__main__':
    print(graph_txt(_vs(),'node'))