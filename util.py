import networkx as nx
import numpy as np
import os

def write_data(directory, filename, data):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    np.save(filepath, data)

def aggregate_multiple_sources(graph, S):
    """
    aggregating multiple seed nodes S to the single seed node s
    
    Input: 
    - graph: a graph (neworkx graph object w/ 'prob' edge attribute) 
    - S: seed nodes (list)
    
    Output:
    - new_graph: a graph with the single node s
    """
    V = list(graph.nodes()) 
    V_minus_S = set(V) - set(S)

    new_node = 's'
    new_graph = graph.copy()
    
    new_edges = []
    for v in V_minus_S:
        prob_tmp = 1.0
        for u in S:
            if new_graph.has_edge(u, v):  
                p_uv = new_graph[u][v]['prob']
                prob_tmp *= (1 - p_uv)
        p_sv = 1 - prob_tmp
        if p_sv > 0:  
            new_edges.append((new_node, v, {"prob": p_sv}))

    new_graph.add_node(new_node)
    new_graph.remove_nodes_from(S)
    new_graph.add_edges_from(new_edges)
    return new_graph, new_node

def inv_log_dijkstra(graph, start):
    """
    calculating the shortest path & distance from a start node to other nodes
    
    Input: 
    - graph: a directed graph w/ edge probability (neworkx graph object) 
    - start: start node
    
    Output:
    - dists: each node's distances from a start node
    - paths: each node's shortest paths from a start node
    """
    new_graph = nx.DiGraph()
    for u, v, data in graph.edges(data=True):
        logp_inv = - np.log(data['prob'])
        new_graph.add_edge(u, v, weight=logp_inv)
    dists, paths = nx.single_source_dijkstra(new_graph, start)
    return dists, paths

### test ###
# graph = nx.DiGraph([(0, 1, {'prob':0.2}), (0, 2, {'prob':0.1}), (1, 2, {'prob':0.8}), (2, 3, {'prob':0.1})])
# start = 0
# d, p = inverse_logarithmic_weight_dijkstra(graph, start)
# print(d)  # Output: {0: 0, 1: 1.6094379124341003, 2: 1.83258146374831, 3: 4.135166556742355}
# print(p)  # Output: {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3]}