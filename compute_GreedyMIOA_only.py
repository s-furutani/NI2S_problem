import random
import time
import numpy as np
import networkx as nx
import importlib
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys

import util
import read_network as netw
import method as mtd
import signal

importlib.reload(util)
importlib.reload(mtd)
importlib.reload(netw)


class TimeoutException(Exception):
	pass

def timeout_handler(signum, frame):
	raise TimeoutException("Intervention nodes computation took too long to complete.")

def update_q(q, epsilon, X):
	q_X = q.copy()
	for v in X:
		q_X[v] = np.min([q_X[v] + epsilon[v], 1])
	return q_X

def propagation_probability(graph, source):
    dist, MIP = util.inv_log_dijkstra(graph, source)
    pp = {node: 0 for node in graph.nodes()}
    for v, dist in dist.items():
        x = 1/np.exp(dist)
        pp[v] = x
    return pp

def set_q(graph, S, cond_name):
    random.seed(42)
    q = {node: random.uniform(0.5, 1) for node in graph.nodes()}
    for s in S:
        q[s] = 1
    if cond_name=='random':
        print('cond: random agreement condition')
        print('')
        pass
    elif cond_name == 'follower':
        delS = set()  # delS: neighbors (successors) of the node set S
        for v in S:
            delS.update(graph.neighbors(v))
        for node in delS:
            q[node] = 1
        print('cond: follower agreement condition')
        print('')
    elif cond_name == 'proximity':
        new_graph, source = util.aggregate_multiple_sources(graph, S)
        pp = propagation_probability(new_graph, source)
        for node in q.keys():
            if node not in S:
                q[node] = 1 / (1 + np.exp(- 100 * pp[node]))
        print('cond: proximity agreement condition')
        print('')
    else:
        print(f'Error: {cond_name} is undefined agreement condition')
        print('')
    return q

def karate_graph():
    graph = nx.karate_club_graph()
    if not nx.is_directed(graph):
        graph = graph.to_directed()
    d_in = graph.in_degree()
    for u, v in graph.edges():
        graph[u][v]['prob'] = 1./d_in[v]
    return graph

def check_property_of_interv_nodes(X, new_graph, new_q):
	d, p = util.inv_log_dijkstra(new_graph, 's')
	for x in X[:5]:
		print('x:', x)
		print('  degree:', new_graph.out_degree(x))
		print('  q:', new_graph.nodes[x]['q'])
		try:
			print('  distance:', d[x])
		except:
			print('  distance: -')

### pre-processing ###
def pre_processing(graph, S, eps):
    new_graph, source = util.aggregate_multiple_sources(graph, S)
    new_graph.nodes['s']['q'] = 1
    new_q = nx.get_node_attributes(new_graph, 'q')
    new_epsilon = {key: eps for key in new_q}
    return new_graph, source, new_q, new_epsilon

def get_random_high_degree_nodes(graph, num_nodes):
    random.seed(42)
    out_degrees = dict(graph.out_degree())
    top_20_nodes = sorted(out_degrees, key=out_degrees.get, reverse=True)[:20]
    S = random.sample(top_20_nodes, num_nodes)
    return S

if len(sys.argv) < 2:
	print("Error: No argument provided. Please specify 'random', 'follower', or 'proximity'.")
	sys.exit(1)

if sys.argv[1] not in ['random', 'follower', 'proximity']:
	print(f"Error: '{sys.argv[1]}' is not a valid argument.")
	sys.exit(1)

cond_name = sys.argv[1]  # 'random', 'follower', or 'proximity'
print('cond:', cond_name)
print('')

graphs = [netw.Facebook_graph(), netw.WikiVote_graph(), netw.LastFM_graph(), netw.HepTh_graph(), netw.cit_HepTh_graph(), netw.Deezer_graph(), netw.Enron_graph(), netw.Epinions_graph(), netw.Twitter_graph()]
graph_names = ['Facebook', 'WikiVote', 'LastFM', 'HepTh', 'cit-HepTh', 'Deezer', 'Enron', 'Epinions', 'Twitter']
kmax = 200
eps = 1.0
theta = 0.0001

for i in range(5):
	### initial setting ###
	graph = graphs[i]
	graph_name = graph_names[i]

	### directory path ###
	directory = 'results/'+graph_name+'/'+cond_name+'/'

	### seed nodes ###
	random.seed(42)
	S = random.sample([node for node in graph.nodes() if graph.degree(node) > 10], 5)
	# S = get_random_high_degree_nodes(graph, num_nodes=5)
	print('****' * 10)
	print('')
	print(f'graph: {graph_name}, seed node: {S}')
	print('')

	### agreement condition ###
	q = set_q(graph, S, cond_name)
	epsilon = {key: eps for key in q}
	# epsilon = {key: 0.2 for key in q}
	nx.set_node_attributes(graph, q, 'q')

	### pre-processing ###
	# convert the input graph to the graph w/ the single seed node & get corresponding q and epsilon
	new_graph, source, new_q, new_epsilon = pre_processing(graph, S, eps)

	### intervening nodes computation ###

	try:
		# For the Deezer, Enron, Epinions, and Twitter graph, GreedyMIOA computation will take >24 hours.
		signal.signal(signal.SIGALRM, timeout_handler)
		signal.alarm(86400)

		print('computing intervening nodes by GreedyMIOA')
		start_time = time.time()
		X = mtd.GreedyMIOA(new_graph, source, new_q, new_epsilon, kmax, theta)
		end_time = time.time()
		t = np.round(end_time - start_time, 2)
		print(f"  {t} sec")
		util.write_data(directory, 'GreedyMIOA', X)
	except TimeoutException as e:
		print(e)
		# break

	# if i < 6:
	# 	# For the Deezer, Enron, Epinions, and Twitter graph, GreedyMIOA computation will take >24 hours.
	# 	print('computing intervening nodes by GreedyMIOA')
	# 	start_time = time.time()
	# 	X = mtd.GreedyMIOA(new_graph, source, new_q, new_epsilon, kmax, theta)
	# 	end_time = time.time()
	# 	t = np.round(end_time - start_time, 2)
	# 	print(f"  {t} sec")
	# 	util.write_data(directory, 'GreedyMIOA', X)
