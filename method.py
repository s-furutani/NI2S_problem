import time
import networkx as nx
import random
import numpy as np
from util import aggregate_multiple_sources, inv_log_dijkstra
from util import inv_log_dijkstra


def BaselineRandom(graph, source, k):
    V_minus_S = list(set(graph.nodes()) - {source})
    X = random.sample(V_minus_S, k)
    return X

def BaselineDegree(graph, source, k):
    degree = sorted(graph.out_degree, key=lambda x: x[1], reverse=True) 
    degree_sorted_nodes = [node for node, _ in degree if node != source]
    X = degree_sorted_nodes[:k]
    return X

def BaselineDistance(graph, source, k):
    dist, path = inv_log_dijkstra(graph, source)
    sorted_indices = sorted(dist, key=dist.get, reverse=False) 
    X = sorted_indices[1:k+1]
    return X

###############################################################
#                   GreedyMIOA & FastMIOA                     #
###############################################################

# compute positive activation probability along with Maximum Influence Path
def compute_pap(q, MIP):
    qq = {key: 0 for key in q}
    for v, path in MIP.items():
        x = 1
        for i in path:
            x *= q[i] 
        qq[v] = x
    return qq

# compute approximate positive influence spread on MIOA for given q
def compute_phi(q, pp, MIP):
    N = len(q)
    qq = compute_pap(q, MIP)
    sigma = 0
    for k in qq.keys():
        sigma +=  pp[k] * qq[k]
    return sigma

def get_top_k_keys(dicts, k):
    top_k_keys = [key for key, value in sorted(dicts.items(), key=lambda item: item[1], reverse=True)[:k]]
    return top_k_keys

def FastMIOA(graph, source, q, epsilon, k, theta):
    """
    calculating the set of intervening nodes by FastMIOA
    
    Input: 
    - graph: a graph with a single seed node
    - source: the seed node
    - q: positive activation probability of each node
    - epsilon: individual intervention effect of each node
    - k: a budget
    - theta: threshold parameter of MIOA

    Output:
    - X: the set of intervention nodes
    """
    dist, MIP = inv_log_dijkstra(graph, source)

    pp = {node: 0 for node in graph.nodes()}
    for v, dist in dist.items():
        x = 1/np.exp(dist)
        if x > theta:
            pp[v] = x
    MIOA_nodes = {k for k, v in pp.items() if v != 0}  # node set in MIOA
    print(len(MIOA_nodes))

    phi = compute_phi(q, pp, MIP)
    diffs = {node: 0 for node in MIOA_nodes}
    for v in MIOA_nodes:
        q_tmp = q.copy()
        q_tmp[v] = min(q[v] + epsilon[v], 1)
        phi_tmp = compute_phi(q_tmp, pp, MIP)
        diffs[v] = phi_tmp - phi
    X = get_top_k_keys(diffs, k)
    return X

def GreedyMIOA(graph, source, q, epsilon, k, theta):
    """
    calculating the set of intervening nodes by GreedyMIOA
    
    Input: 
    - graph: a graph with a single seed node
    - source: the seed node
    - q: positive activation probability of each node
    - epsilon: individual intervention effect of each node
    - k: a budget
    - theta: threshold parameter of MIOA

    Output:
    - X: the set of intervention nodes
    """
    dist, MIP = inv_log_dijkstra(graph, source)

    pp = {node: 0 for node in graph.nodes()}
    for v, dist in dist.items():
        x = 1/np.exp(dist)
        if x > theta:
            pp[v] = x
    MIOA_nodes = {k for k, v in pp.items() if v != 0}  # node set in MIOA
    print(len(MIOA_nodes))

    phi = compute_phi(q, pp, MIP)
    X = []  # Intervention nodes set

    q_copy = q.copy()

    for l in range(k):
        diffs = {node: 0 for node in MIOA_nodes if node not in X}
        for v in diffs:
            q_tmp = q_copy.copy()
            q_tmp[v] = min(q_copy[v] + epsilon[v], 1)
            phi_tmp = compute_phi(q_tmp, pp, MIP)
            diffs[v] = phi_tmp - phi

        best_v = max(diffs, key=diffs.get)
        X.append(best_v)

        q_copy[best_v] = min(q_copy[best_v] + epsilon[best_v], 1)
        phi = compute_phi(q_copy, pp, MIP)  # 更新後のphiを計算
    return X

######################################################
#                   AdvancedGreedy                   #
######################################################

def sampled_graph(graph):
    """
    sampling a live-edge graph
    """
    g = nx.create_empty_copy(graph)
    for u, v, data in graph.edges(data=True):
        p = data.get('prob', 1.0)  # default 1.0 (if no "prob" attribute)
        if random.random() <= p:  # keeping the edge e with probability p(e)
            g.add_edge(u, v, **data)
    return g

def dominator_tree(graph, source):
    '''
    constructing the dominator tree of a graph with a root (source) node
    '''
    if not nx.is_directed(graph):
        graph = graph.to_directed()
    DT_edges_r = nx.immediate_dominators(graph, source).items()  # {(u, idom(u))}
    DT_edges = [(y, x) for x, y in DT_edges_r]  # {(idom(u), u)}
    DT = nx.DiGraph(DT_edges)
    DT.remove_edges_from(list(nx.selfloop_edges(DT)))
    return DT

def subtree_size(tree, u):
    '''
    calculating the subtree size when node u is the root
    '''
    def counting(v):
        # counting itself
        count = 1 
        # counting recursively for all child nodes
        for child in tree.successors(v):
            count += counting(child)
        return count
    return counting(u)        

def compute_ESD(graph, source, num_samples):
    """
    computing Expected Spread Decrease (ESD) for each u ∈ V-{s}    
    
    Input: 
    - graph: a graph with a single seed node
    - source: seed node
    - num_samples: # of sampled graphs

    Output:
    - expected spread decrease
    """
    
    ESD = {node: 0 for node in graph.nodes()}
    for _ in range(num_samples):
        g = sampled_graph(graph)  # generate a sampled graph
        DT = dominator_tree(g, source)  # construct the domniator tree of g
        # count the size of subtree in DT when each node u is the root
        c = {}
        for node in graph.nodes():
            try:
                c[node] = subtree_size(DT, node)
            except Exception as e:
                #print(e)
                c[node] = 0         
        # compute ESD for each node u
        for u in graph.nodes():
            ESD[u] += c[u] / num_samples
    ESD[source] = -999  # process for avoiding source node selection
    return ESD


def AdvancedGreedy(graph, source, k, num_samples):
    """
    calculating the blocker set by AdvancedGreedy    
    
    Input: 
    - graph: a graph with a single seed node
    - source: the seed node
    - k: a budget
    - num_samples: # of sampled graphs

    Output:
    - B: the blocker set
    """
    nodes = graph.nodes()
    B = []
    for _ in range(k):
        NB = list(set(nodes) - set(B))
        graph_NB = graph.subgraph(NB)
        ESD = compute_ESD(graph_NB, source, num_samples)
        x = source  # Note: ESD[source] = -999
        for u in graph_NB.nodes():
            if ESD[u] > ESD[x]:
                x = u
        B.append(x)
    return B

def GreedyReplace(graph, source, k, num_samples):
    """
    calculating the blocker set by GreedyReplace
    
    Input: 
    - graph: a graph with a single seed node
    - source: the seed node
    - k: a budget
    - num_samples: # of sampled graphs

    Output:
    - B: the blocker set
    """
    nodes = graph.nodes()
    CB = list(graph.successors(source))
    deg_s = len(CB)
    B = []
    for _ in range(min(deg_s, k)):
        NB = list(set(nodes) - set(B))
        graph_NB = graph.subgraph(NB)
        ESD = compute_ESD(graph_NB, source, num_samples)
        x = source  # Note: ESD[source] = -999
        for u in CB:
            if ESD[u] > ESD[x]:
                x = u
        CB.remove(x)
        B.append(x)
    reversed_B = list(reversed(B))
    for v in reversed_B:
        B.remove(v)
        NB = list(set(nodes) - set(B))
        graph_NB = graph.subgraph(NB)
        ESD = compute_ESD(graph_NB, source, num_samples)
        x = source
        for u in graph_NB.nodes():
            if ESD[u] > ESD[x]:
                x = u
        B.append(x)
        if v == x:
            break
    return B

######################################################
#                   BasicGreedy                      #
######################################################

def run_ICN(graph, S, q):
    """
    run IC-N model
    
    Input: 
    - graph: a graph (networkx object w/ "prob" edge attribute)
    - S: the set of seed nodes
    - q: positive activation probability of each node
    
    Output: 
    - fin_posi: final positively activated nodes
    - fin_nega: final negatively activated nodes
    """
    node_status = {node: 0 for node in graph.nodes()}  # node activation status (0: inactive, 1: positive, 2: negative) 
    for s in S:
        node_status[s] = 1
    A = list(S.copy())  # current activated nodes
    while len(A) > 0:
        #print('current node status:', node_status)
        #print('current activated nodes:', A)
        #print('')
        newA = []
        for u in A:
            neighbors = list(graph.neighbors(u))
            for v in neighbors:
                if node_status[v] == 0 and random.random() < graph[u][v]['prob']:
                    newA.append(v)
                    # if node u is negative, node v becomes negative
                    if node_status[u] == 2:
                        node_status[v] = 2
                    # if node u is positive, node v becomes positive with prob. q_v or negative with prob. (1 - q_v)
                    else:
                        if random.random() < q[v]:
                            node_status[v] = 1
                        else:
                            node_status[v] = 2
        A = newA
        random.shuffle(A)
    fin_posi = {node for node, stat in node_status.items() if stat == 1} # finally positively activated nodes; fin_posi = {v ∈ V | node_status[v] = 1} 
    fin_nega = {node for node, stat in node_status.items() if stat == 2} # finally negatively activated nodes; fin_nega = {v ∈ V | node_status[v] = 2} 
    # print(fin_posi)
    # print(fin_nega)
    # print('')
    return fin_posi, fin_nega

### run IC-N simulation ###
def ICN_simulation(graph, S, q, num_simulation):
    Z = []
    for _ in range(num_simulation):
        fin_p, fin_n = run_ICN(graph, S, q)
        z = len(fin_n) / (len(fin_p) + len(fin_n))
        Z.append(z)
    ave = np.mean(Z)
    ci = 1.96 * np.std(Z) / num_simulation  # 95% confidence interval
    return ave, ci

def BasicGreedy(graph, S, q, epsilon, k, num_samples=10):
    """
    calculating the set of intervening nodes by BasicGreedy
    
    Input: 
    - graph: a graph with a single seed node
    - source: the seed node
    - k: a budget
    - num_samples: # of sampled graphs

    Output:
    - B: the blocker set
    """
    V = list(graph.nodes())
    X = []
    q_copy = q.copy()
    sigma, _ = ICN_simulation(graph, S, q_copy, num_samples)
    for l in range(k):
        
        start_time = time.time()

        diffs = {node: 0 for node in V if node not in X}
        for v in diffs:
            q_tmp = q_copy.copy()
            q_tmp[v] = min(q_copy[v] + epsilon[v], 1)
            sigma_tmp, _ = ICN_simulation(graph, S, q_copy, num_samples)
            diffs[v] = sigma_tmp - sigma

        best_v = max(diffs, key=diffs.get)
        X.append(best_v)

        q_copy[best_v] = min(q_copy[best_v] + epsilon[best_v], 1)
        sigma, _ = ICN_simulation(graph, S, q_copy, num_samples)
        
        end_time = time.time()
        t = np.round(end_time - start_time, 2)
        print('l:', t)

    return X

