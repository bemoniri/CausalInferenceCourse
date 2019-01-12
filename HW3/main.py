import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, permutations

#%%
def getChainComp (G):
    Adj = nx.adjacency_matrix(G)
    Adj = Adj.todense()
    sizeofmat = np.shape(Adj)
    for i in range(sizeofmat[0]):
        for j in range(sizeofmat[1]):
            if Adj[i,j] == 1 and Adj[j,i] ==0:
                Adj[i,j] = 0
    Gp = nx.from_numpy_matrix(Adj)
    O = []
    for g in nx.connected_component_subgraphs(Gp):
        O.append(g)
    return O    


#%%
def ChainCom(U, v):
    A = [v]
    B = [x for x in U.nodes() if x!=v]
    G = U
    while (len(B) != 0):        
        AdjA = []
        for y in A:
            w = G.neighbors(y)
            w = [x for x in w]
            AdjA = AdjA + w
        T = [w for w in B if w in AdjA]   
        for t in T:
            for c in A:
                if [t, c] in G.edges():
                    G.remove_edge(t, c)
        while(True):
            flag = 0
            GT = G.subgraph(T)
            GTedges = [x for x in GT.edges()]            
            for (y, z) in GTedges:                
                if (y, z) in G.edges():
                    if (z, y) in G.edges():     
                        for x in G.node():
                            if (x, z) in G.edges() and (z, x) not in G.edges():        
                                if (x, y) not in G.edges() and (y, x) not in G.edges():                                
                                  if (y,z) in G.edges():
                                      G.remove_edge(y, z)                              
                                      flag = 1                                      
            if(flag == 0):
                A = T
                B = [x for x in B if x not in T]
                flag  = 0            
                break
    O = getChainComp(G)    
    return [G, O]

#%% Testing Everything until now

nodes = [0, 1, 2, 3, 4]
GG = nx.DiGraph()
GG.add_nodes_from(nodes)
GG.add_edges_from([[1, 2], [2, 1], [1, 4], [4,1], [0, 1],[1, 0], [0, 2], [2,0], [2, 4], [4, 2], [1, 3], [3, 1], [3, 4], [4,3]])
Adj = nx.adjacency_matrix(GG)
Adj = Adj.todense()


plt.figure()
plt.subplot(2, 3, 1)
nx.draw_shell(GG,  with_labels=True)
plt.title('UCCG')
c = 2
for j in nodes:
    H = nx.to_directed(nx.from_numpy_array(Adj))
    [G, O] = ChainCom(H, j)
    plt.subplot(2, 3, c)
    nx.draw_shell(G, with_labels=True)
    plt.title("%i-rooted" %(c-1))
    c = c + 1
plt.show()

#%%
def sizeMEC (U):
    n = U.number_of_edges()
    p = U.number_of_nodes()
    if (n==p-1):
        return p
    if (n==p):
        return 2*p
    if (n==p*(p-1)/2 -2):
        return (p**2 - p - 4)*np.math.factorial(p-3)
    if (n==p*(p-1)/2 -1):
        return 2*np.math.factorial(p-1) - np.math.factorial(p-2)
    if (n==p*(p-1)/2):    
        return np.math.factorial(p)    
    Adj = nx.adjacency_matrix(U)
    Adj = Adj.todense()    
    s = [i for i in range(p)]
    for j in range(p):
        H = nx.to_directed(nx.from_numpy_array(Adj))
        [G, O] = ChainCom(H, j)        
        s[j] = 1
        for graph in O:            
            s[j] = s[j] * sizeMEC(graph)
    return sum(s)


#%% Test Counting    
nodes = [0, 1, 2, 3, 4]
GG = nx.DiGraph()
GG.add_nodes_from(nodes)
GG.add_edges_from([[1, 2], [2, 1], [1, 4], [4,1], [0, 1], [1, 0], [0, 2], [2,0], [2, 4], [4, 2], [1, 3], [3, 1], [3, 4], [4,3]])
print(sizeMEC(GG))


#%% Uniform MEC Sampling
def randomDAG(n, p):
    nodes = range(n)
    Graph = nx.DiGraph()
    Graph.add_nodes_from(nodes)
    for possible_edges in combinations(nodes, 2):
        if (np.random.rand() < p):
            Graph.add_edge(possible_edges[0], possible_edges[1])    
    return Graph


def _has_both_edges(dag, i, j):
    return dag.has_edge(i, j) and dag.has_edge(j, i)

def _has_any_edge(dag, i, j):
    return dag.has_edge(i, j) or dag.has_edge(j, i)

def _has_one_edge(dag, i, j):
    return ((dag.has_edge(i, j) and (not dag.has_edge(j, i))) or
            (not dag.has_edge(i, j)) and dag.has_edge(j, i))
def _has_no_edge(dag, i, j):
    return (not dag.has_edge(i, j)) and (not dag.has_edge(j, i))




def dag2cpdag(dag):    
    vstructure_set = []
    for node in dag.nodes():        
        for edge1 in dag.edges():
            if(edge1[1] == node):
                for edge2 in dag.edges():
                    if(edge2[1] == node and edge2 != edge1):
                        if([edge2[0], edge1[0]] not in dag.edges() and [edge1[0], edge2[0]] not in dag.edges()):
                            if ([node, edge1[0], edge2[0]] not in vstructure_set):
                                vstructure_set.append([node, edge2[0], edge1[0]])                                    
    cpdag = nx.DiGraph()
    cpdag.add_nodes_from(dag.node())    
    for edge in dag.edges():
        cpdag.add_edge(edge[0], edge[1])
        cpdag.add_edge(edge[1], edge[0])    
        
    for vstructure in vstructure_set:
        if ((vstructure[0], vstructure[2]) in cpdag.edges()):
            cpdag.remove_edge(vstructure[0], vstructure[2])        
        if ((vstructure[0], vstructure[1]) in cpdag.edges()):
            cpdag.remove_edge(vstructure[0], vstructure[1])            
    # MEEK RULES!!! ::
    node_ids = dag.nodes()
    dag = cpdag
    old_dag = dag.copy()
    while True:
        for (i, j) in permutations(node_ids, 2):            
            # RULE 1
            if _has_both_edges(dag, i, j): 
                for k in dag.predecessors(i):
                    if dag.has_edge(i, k):
                        continue
                    if _has_any_edge(dag, k, j):
                        continue
                    dag.remove_edge(j, i)
                    break    
            # RULE 2
            if _has_both_edges(dag, i, j):
                succs_i = set()
                for k in dag.successors(i):
                    if not dag.has_edge(k, i):
                        succs_i.add(k)
                preds_j = set()
                for k in dag.predecessors(j):
                    if not dag.has_edge(j, k):
                        preds_j.add(k)
                if len(succs_i & preds_j) > 0:
                    dag.remove_edge(j, i)
            # RULE 3
            if _has_both_edges(dag, i, j):
                adj_i = set()
                for k in dag.successors(i):
                    if dag.has_edge(k, i):
                        adj_i.add(k)
                for (k, l) in combinations(adj_i, 2):
                    if _has_any_edge(dag, k, l):
                        continue
                    if dag.has_edge(j, k) or (not dag.has_edge(k, j)):
                        continue
                    if dag.has_edge(j, l) or (not dag.has_edge(l, j)):
                        continue
                    dag.remove_edge(j, i)
                    break
            # Rule 4: Orient i-j into i->j whenever there are two chains
            # i-k->l and k->l->j such that k and j are nonadjacent.            
            # However, this rule is not necessary when the PC-algorithm
            # is used to estimate a DAG.
        if nx.is_isomorphic(dag, old_dag):
            break
        old_dag = dag.copy()        
    return dag


#%%
def CountMEC (cpdag):
    graphs = getChainComp(cpdag)
    counter = 1
    for uccg in graphs:
        counter = counter * sizeMEC(uccg)        
    return counter


def AdjCountMEC(adj):
    cpdag = nx.from_numpy_matrix(adj)
    graphs = getChainComp(cpdag)
    counter = 1
    for uccg in graphs:
        counter = counter * sizeMEC(uccg)        
    return counter

#%% Testing Meek Rules and DAG2CPDAG Function
nodes = [0, 1, 2, 3, 4]
GG = nx.DiGraph()
GG.add_nodes_from(nodes)
GG.add_edges_from([[1,0], [1, 2], [2, 0], [3,2], [3, 4]])

cpdag = dag2cpdag(GG)


plt.figure()
plt.subplot(121)
nx.draw_shell(GG, with_labels=True)
plt.title("DAG")
plt.subplot(122)
nx.draw_shell(cpdag, with_labels=True)
plt.title("CPDAG")
plt.show()

#%% Check "Counting Markov Equivalenece Class" Function
print(CountMEC(cpdag))

#%% Random Dag
h = []
for i in range(10000): 
    print(i)
    rdag = randomDAG(11, 0.5)
    cpdag = dag2cpdag(rdag)    
    h.append([CountMEC(cpdag), rdag.number_of_edges()])
numDags = 4175098976430598143

plt.figure()
plt.scatter([x[0] for x in h], [x[1] for x in h])
plt.show()
