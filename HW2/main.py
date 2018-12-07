import numpy as np
import pandas as pd
import csv
import scipy
from itertools import combinations, permutations
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import norm


#%% Partial Correlation Calculations and The Cond. Independence Test
def test (data, x, y, set):
    if (len(set) == 0):
        data_x = data[:, x]
        data_y = data[:, y]
        
        r = np.corrcoef(data_x, data_y)
        r = r[1,0]
        t = 0.5*np.log((1+r)/(1-r))   
        n = len(data_x)        
        z = t*np.sqrt(n-3)
        pval = scipy.stats.norm.sf(abs(z))*2 #twosided
        return pval

    else:
        data_x = data[:, x]
        data_y = data[:, y]
        j = 0
        data_z = np.zeros([len(data_x), len(set)])
        for i in set:
            data_z[:, j] = data[:, i]
            j += 1
        reg = LinearRegression().fit(data_z, data_x)
        pred_x = reg.predict(data_z)
        residual_x = pred_x - data_x
                
        reg = LinearRegression().fit(data_z, data_y)
        pred_y = reg.predict(data_z)
        residual_y = data_y - pred_y
                
        r = np.corrcoef(residual_x, residual_y)
        r = r[1,0]
        t = 0.5*np.log((1+r)/(1-r))   
        n = len(data_x)        
        z = t*np.sqrt(n-3)
        pval = scipy.stats.norm.sf(abs(z))*2 #twosided
        return pval
        
#%% PC Algorithm Implementation
def pc (da, alpha):
    # Nodes of the graph
    b = np.shape(da)
    nodes = range(b[1])
    # Creating the full undirected graph to start with:
    Graph = nx.Graph()
    Graph.add_nodes_from(nodes)
    for (i, j) in combinations(nodes, 2):
        Graph.add_edge(i, j)
        pass
    l = -1
    while True:
        l = l + 1
        for (i, j) in permutations(nodes, 2):
            adjacents = list(Graph.neighbors(i))
            if j in adjacents: # if they are adjacent
                adjacents.remove(j) 
                if len(adjacents)>=l: #if there exists a set of adjacent nodes with size >= l
                    for S in combinations(adjacents, l):
                        pval = test(da, i, j, S)                    # Cond. Ind. Test
                        print("(%d, %d), S = %s,  pval = %f" %(i, j, str(S), pval))                    
                        if pval > alpha:
                            Graph.remove_edge(i, j)   # Remove Edge (PC Algorithm)
                            print("Remove (%d %d)" %(i,j))
                            break
        print("End of Level %d" %l)
        if(Graph.number_of_nodes()<l):
            break
    return Graph

#%% ٍExample One - Testing The Algorithm

x = np.random.randn(1, 10000000)
y = x + np.random.randn(1, 10000000)
z = 1*y + np.random.randn(1, 10000000)
w = 2*z + 3*x +  np.random.randn(1, 10000000)

da = np.transpose((np.concatenate((x, y, z, w), axis =0)))
G = pc(da, 0.02)        

plt.figure()
nx.draw_networkx(G)
plt.show()


#%% ٍExample Two - Testing The Algorithm

x0 = np.random.randn(1, 10000000)
x1 =  x0 + np.random.randn(1, 10000000)
x2 =  2*x0 + x1 +  0.01*np.random.randn(1, 10000000)
x3 =  3*x1 + 0.01*np.random.randn(1, 10000000)
x4 =  4*x2 + 10*x3 + 2*np.random.randn(1, 10000000)

da = np.transpose((np.concatenate((x0, x1, x2, x3, x4), axis =0)))

G = pc(da, 0.02)      
plt.figure()
nx.draw_networkx(G)
plt.show()

#%% pc-data.csv

# Loading Data
dag_data = np.genfromtxt('pc-data.csv', delimiter='\t')
# PC-pop Algorithm:
G = pc(dag_data, 0.02)
print(G.number_of_edges())


#%% PC-stable Algorithm implementation
def pc_stable (da, alpha):
    remove_edges = []
    # Nodes of the graph
    b = np.shape(da)
    nodes = range(b[1])
    # Creating the full undirected graph to start with:
    Graph = nx.Graph()
    Graph.add_nodes_from(nodes)
    for (i, j) in combinations(nodes, 2):
        Graph.add_edge(i, j)
        pass
    l = 0
    while True:
        for (i, j) in permutations(nodes, 2):
            adjacents = list(Graph.neighbors(i))
            if j in adjacents: # if they are adjacent
                adjacents.remove(j) 
                if len(adjacents)>=l: #if there exists a set of adjacent nodes with size >= l
                    for S in combinations(adjacents, l):
                        pval = test(da, i, j, S)                    # Cond. Ind. Test
                        print("(%d, %d), S = %s,  pval = %f" %(i, j, str(S), pval))                    
                        if pval > alpha:
                            remove_edges.append((i, j))
                            
                            print("(%d %d) will be removed" %(i,j))
                            break
        l = l + 1
        Graph.remove_edges_from(remove_edges) 
        print("End of Level %d, Edges Removed!" %l)
        if(Graph.number_of_nodes()<l):
            break
    return Graph


#%% ٍExample - Testing The Algorithm (PC-Stable)

x = np.random.randn(1, 10000000)
y = x + np.random.randn(1, 10000000)
z = 1*y + x + np.random.randn(1, 10000000)
w = 2*z + 3*x +  np.random.randn(1, 10000000)

da = np.transpose((np.concatenate((x, y, z, w), axis =0)))
G = pc_stable(da, 0.02)        

plt.figure()
nx.draw_networkx(G)
plt.show()


#%% pc-data.csv

# Loading Data
dag_data = np.genfromtxt('pc-data.csv', delimiter='\t')
# PC-pop Algorithm:
G = pc_stable(dag_data, 0.02)
print(G.number_of_edges())

plt.figure()
nx.draw_kamada_kawai(G)
plt.show()

#%%
def randomDAG(n, p, lB, uB):
    nodes = range(n)
    G = nx.Graph()
    Graph = nx.Graph()
    Graph.add_nodes_from(nodes)
    for possible_edges in combinations(nodes, 2):
        if (np.random.rand() < p):
            Graph.add_edge(possible_edges[0], possible_edges[1], weight=np.random.uniform(lB, uB))    
    plt.figure()
    nx.draw_circular(Graph)
    plt.show()
    return Graph

def genData(Graph, n):
    x = np.zeros((n, Graph.number_of_nodes()))
    print(Graph.nodes())
    for node in Graph.nodes():
        x[:,node] = np.random.randn(n)
        adjacents = Graph.neighbors(node)
        for i in adjacents:
            if i < node: # if parent
                w = Graph.get_edge_data(i, node)
                w = w['weight']
                x[:,node] = x[:,node] + w*x[:,i]
    return x



#%%  PC Alg
Recall_final = []
Missing_final = []
alpha_list = []
for i in [-4, -3, -2, -1, 0, 1, 2]:
    alpha_list.append(0.01*2**i)

for alpha in alpha_list:
    Missing_array = []
    Recall_array = []
    for i in range(30):    
        Graph = randomDAG(20, 0.2, 0.1, 1)
        data = genData(Graph, 1000)
        estimated = pc(data, alpha)
        
        real_adj = nx.to_numpy_matrix(Graph, weight='None')
        estimated_adj = nx.to_numpy_matrix(estimated)
        
        real_num_edges = Graph.number_of_edges()
        
        # Recall : Number of real edges estimated correctly/ Total number of real edges
        delta = 2*real_adj - estimated_adj
        
        edge_in_real_and_in_estimated = np.count_nonzero(delta==1)/2
        recall = edge_in_real_and_in_estimated/real_num_edges
        Recall_array.append(recall)
        
        # Missing : Edges in the estimsted graph which do not exist in the real graph / number of edges in the real graph
        edge_not_in_real_and_in_estimated = np.count_nonzero(delta==-1)/2
        missing = edge_not_in_real_and_in_estimated/real_num_edges
        Missing_array.append(missing)
        
    Recall_final.append(np.mean(Recall_array))
    Missing_final.append(np.mean(Missing_array))

pc_recall = Recall_final
pc_missing = Missing_final
#%%  PC-Stable Alg
Recall_final = []
Missing_final = []
alpha_list = []
for i in [-4, -3, -2, -1, 0, 1, 2]:
    alpha_list.append(0.01*2**i)

for alpha in alpha_list:
    Missing_array = []
    Recall_array = []
    for i in range(30):    
        Graph = randomDAG(20, 0.2, 0.1, 1)
        data = genData(Graph, 1000)
        estimated = pc_stable(data, alpha)
        
        real_adj = nx.to_numpy_matrix(Graph, weight='None')
        estimated_adj = nx.to_numpy_matrix(estimated)
        
        real_num_edges = Graph.number_of_edges()
        
        # Recall : Number of real edges estimated correctly/ Total number of real edges
        delta = 2*real_adj - estimated_adj
        
        edge_in_real_and_in_estimated = np.count_nonzero(delta==1)/2
        recall = edge_in_real_and_in_estimated/real_num_edges
        Recall_array.append(recall)
        
        # Missing : Edges in the estimsted graph which do not exist in the real graph / number of edges in the real graph
        edge_not_in_real_and_in_estimated = np.count_nonzero(delta==-1)/2
        missing = edge_not_in_real_and_in_estimated/real_num_edges
        Missing_array.append(missing)
    Recall_final.append(np.mean(Recall_array))
    Missing_final.append(np.mean(Missing_array))

stable_recall = Recall_final
stable_missing = Missing_final

#%%
pow = [-4,-3,-2,-1,0,1,2]
plt.figure()
plt.plot(pow, pc_recall)
plt.plot(pow, stable_recall)
plt.show()

plt.figure()
plt.plot(pow, pc_missing)
plt.plot(pow, stable_missing)
plt.show()
