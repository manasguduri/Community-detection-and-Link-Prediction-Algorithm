from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import urllib.request

def read_graph():
    """
    Create the example graph from class. Used for testing.
    Do not modify.
    """
    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def bfs(graph, root, max_depth):
    """
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
    E.g., if max_depth=2, only paths of length 2 or less will be considered.
    This means that nodes greather than max_depth distance from the root will not
    appear in the result.
    You may use these two classes to help with this implementation:
      https://docs.python.org/3.5/library/collections.html#collections.defaultdict
      https://docs.python.org/3.5/library/collections.html#collections.deque
    Params:
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.
    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    In the doctests below, we first try with max_depth=5, then max_depth=2.
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> sorted(node2distances.items())
    [('A', 3), ('B', 2), ('C', 3), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('A', 1), ('B', 1), ('C', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('A', ['B']), ('B', ['D']), ('C', ['B']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 2)
    >>> sorted(node2distances.items())
    [('B', 2), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('B', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('B', ['D']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    """
    adj = {}
    node2distances ={}
    node2num_paths = {}
    node2parents = {}
    
    for x in graph.adjacency_iter():
        adj[x[0]] = []
        for n in x[1].keys():
            adj[x[0]].append(n)

    paths = {}
    
    for end in graph.nodes():
        depth = 0
        timeToDepthIncrease = 1
        pendingDepthIncrease = False
    
        # maintain a queue of paths
        queue = []
        # push the first path into the queue
        queue.append([root])
        while queue:
            # get the first path from the queue
            path = queue.pop(0)
            timeToDepthIncrease  = timeToDepthIncrease - 1

            if timeToDepthIncrease  == 0:
                depth = depth + 1
                pendingDepthIncrease = True
        
            # get the last node from the path
            node = path[-1]
            # path found
            if node == end:
                if end not in paths:
                    paths[end] = path
                    node2distances[end] = len(path) - 1
                    node2num_paths[end] = 1
                    node2parents[end] = [path[len(path) - 2]]
                elif(len(path) <= len(paths[end])):
                    node2num_paths[end] = node2num_paths[end] + 1
                    if not path[len(path) - 2] in node2parents[end]:
                        node2parents[end].append(path[len(path) - 2])
                                            
                    
            if depth > max_depth:
                break
        
            # enumerate all adjacent nodes, construct a new path and push it into the queue
            for adjacent in adj[node]:
                new_path = list(path)
                new_path.append(adjacent)
                queue.append(new_path)

            if pendingDepthIncrease:
                timeToDepthIncrease = len(queue)
                pendingDepthIncrease = False

    return node2distances,node2num_paths, node2parents
        

def bottom_up(root, node2distances, node2num_paths, node2parents):
    """
    Compute the final step of the Girvan-Newman algorithm.
    See p 352 From your text:
    https://github.com/iit-cs579/main/blob/master/read/lru-10.pdf
        The third and final step is to calculate for each edge e the sum
        over all nodes Y of the fraction of shortest paths from the root
        X to Y that go through e. This calculation involves computing this
        sum for both nodes and edges, from the bottom. Each node other 
        than the root is given a credit of 1, representing the shortest
        path to that node. This credit may be divided among nodes and
        edges above, since there could be several different shortest paths
        to the node. The rules for the calculation are as follows: ...
    Params:
      root.............The root node in the search graph (a string). We are computing
                       shortest paths from this node to all others.
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    Returns:
      A dict mapping edges to credit value. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).
      Any edges excluded from the results in bfs should also be exluded here.
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> result = bottom_up('E', node2distances, node2num_paths, node2parents)
    >>> sorted(result.items())
    [(('A', 'B'), 1.0), (('B', 'C'), 1.0), (('B', 'D'), 3.0), (('D', 'E'), 4.5), (('D', 'G'), 0.5), (('E', 'F'), 1.5), (('F', 'G'), 0.5)]
    """
    q = deque('')
    credit_dict = {}
    
    for node in node2distances:
        credit_dict[node]=1 if(node!=root)else 0
            
    for n in node2distances:
        flag=True
        if(root!=n):
            for n1 in node2parents.values():
                for value in n1:
                    if(n==value):
                        flag=False
            if(flag):
                q.append(n)
        
    while(len(q)>0):
        current=q.pop()
        for node1 in node2parents[current]:
            tup=()
            if(current<node1):
                tup=current,node1
            else:
                tup=node1,current
            credit_dict[tup]=(credit_dict[current]/node2num_paths[current])
            if node1!=root:
                credit_dict[node1]+=credit_dict[tup]
            if node1 not in q and root!=node1:      
                q.appendleft(node1)
            
    result={}
    for a in credit_dict:
        if(type(a)==tuple):
            result[a]=credit_dict[a]          
    return result

def approximate_betweenness(graph, max_depth):
    """
    Compute the approximate betweenness of each edge, using max_depth to reduce
    computation time in breadth-first search.
    You should call the bfs and bottom_up functions defined above for each node
    in the graph, and sum together the results. Be sure to divide by 2 at the
    end to get the final betweenness.
    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.
    Returns:
      A dict mapping edges to betweenness. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).
    >>> sorted(approximate_betweenness(example_graph(), 2).items())
    [(('A', 'B'), 2.0), (('A', 'C'), 1.0), (('B', 'C'), 2.0), (('B', 'D'), 6.0), (('D', 'E'), 2.5), (('D', 'F'), 2.0), (('D', 'G'), 2.5), (('E', 'F'), 1.5), (('F', 'G'), 1.5)]
   """
    new_dict = {}
    x = defaultdict(list)
    for root in graph.nodes():
        
        node2distances,node2num_paths, node2parents=bfs(graph, root, max_depth)
        s = bottom_up(root, node2distances, node2num_paths, node2parents)
        #print(s)
        for word in s:
            x[word].append(s[word])
            
    for k, v in x.items():
        new_dict[k] = sum(v)/2    

    return new_dict
        
def partition_girvan_newman(graph, max_depth):
    """
    Use your approximate_betweenness implementation to partition a graph.
    Unlike in class, here you will not implement this recursively. Instead,
    just remove edges until more than one component is created, then return
    those components.
    That is, compute the approximate betweenness of all edges, and remove
    them until multiple comonents are created.
    You only need to compute the betweenness once.
    If there are ties in edge betweenness, break by edge name (e.g.,
    (('A', 'B'), 1.0) comes before (('B', 'C'), 1.0)).
    Note: the original graph variable should not be modified. Instead,
    make a copy of the original graph prior to removing edges.
    See the Graph.copy method https://networkx.github.io/documentation/development/reference/generated/networkx.Graph.copy.html
    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.
    Returns:
      A list of networkx Graph objects, one per partition.
    >>> components = partition_girvan_newman(example_graph(), 5)
    >>> components = sorted(components, key=lambda x: sorted(x.nodes())[0])
    >>> sorted(components[0].nodes())
    ['A', 'B', 'C']
    >>> sorted(components[1].nodes())
    ['D', 'E', 'F', 'G']
    """   

    if graph.order() == 1:
        return [graph.nodes()]
    
    
    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)
        # eb is dict of (edge, score) pairs, where higher is better
        # Return the edge with the highest score.
        return sorted(eb.items(), key=lambda x: x[1], reverse=True)

    edge_to_remove = find_best_edge(graph)
    #     print edge_to_remove[0][0]
    # Each component is a separate community. We cluster each of these.
    components = [c for c in nx.connected_component_subgraphs(graph)]
    #     indent = '   ' * depth  # for printing
    #     while len(components) == 1:
    #         print indent + 'removing', edge_to_remove
    for i in range(len(edge_to_remove)):
        graph.remove_edge(*edge_to_remove[i][0])
        components = [c for c in nx.connected_component_subgraphs(graph)]
        if len(components)!=1:
            break    
    return components

def get_subgraph(graph, min_degree):
    """Return a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    We'll use this in the main method to prune the original graph.
    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.
    >>> subgraph = get_subgraph(example_graph(), 3)
    >>> sorted(subgraph.nodes())
    ['B', 'D', 'F']
    >>> len(subgraph.edges())
    2
    """
    new_nodes = []
    for node in graph.nodes():
        degree =graph.degree(nbunch=node)
        if degree >= min_degree:
            new_nodes.append(node)
    k = graph.subgraph(new_nodes)
    
    return k


""""
Compute the normalized cut for each discovered cluster.
I've broken this down into the three next methods.
"""

def volume(nodes, graph):
    """
    Compute the volume for a list of nodes, which
    is the number of edges in `graph` with at least one end in
    nodes.
    Params:
      nodes...a list of strings for the nodes to compute the volume of.
      graph...a networkx graph
    >>> volume(['A', 'B', 'C'], example_graph())
    4
    """
    return len(graph.edges(nodes))    


def cut(S, T, graph):
    """
    Compute the cut-set of the cut (S,T), which is
    the set of edges that have one endpoint in S and
    the other in T.
    Params:
      S.......set of nodes in first subset
      T.......set of nodes in second subset
      graph...networkx graph
    Returns:
      An int representing the cut-set.
    >>> cut(['A', 'B', 'C'], ['D', 'E', 'F', 'G'], example_graph())
    1
    """
    L = [(x,y) for x in S for y in T if graph.has_edge(x,y)]
    return len(L)


def norm_cut(S, T, graph):
    """
    The normalized cut value for the cut S/T. (See lec06.)
    Params:
      S.......set of nodes in first subset
      T.......set of nodes in second subset
      graph...networkx graph
    Returns:
      An float representing the normalized cut value
    """
    cutx = cut(S,T,graph)
    vol1 = float(volume(S,graph))
    vol2 = float(volume(T,graph))
    norm_cut_val = (cutx/vol1)+(cutx/vol2)
    return norm_cut_val


def score_max_depths(graph, max_depths):
    """
    In order to assess the quality of the approximate partitioning method
    we've developed, we will run it with different values for max_depth
    and see how it affects the norm_cut score of the resulting partitions.
    Recall that smaller norm_cut scores correspond to better partitions.
    Params:
      graph........a networkx Graph
      max_depths...a list of ints for the max_depth values to be passed
                   to calls to partition_girvan_newman
    Returns:
      A list of (int, float) tuples representing the max_depth and the
      norm_cut value obtained by the partitions returned by
      partition_girvan_newman. See Log.txt for an example.
    """
    score = []
    for i in max_depths:
        partition_girvan = partition_girvan_newman(graph, i)
        norm = norm_cut(S, T, graph)
        score.append((i, norm_cut))
    return score

def make_training_graph(graph, test_node, n):
    """
    To make a training graph, we need to remove n edges from the graph.
    As in lecture, we'll assume there is a test_node for which we will
    remove some edges. Remove the edges to the first n neighbors of
    test_node, where the neighbors are sorted alphabetically.
    E.g., if 'A' has neighbors 'B' and 'C', and n=1, then the edge
    ('A', 'B') will be removed.
    Be sure to *copy* the input graph prior to removing edges.
    Params:
      graph.......a networkx Graph
      test_node...a string representing one node in the graph whose
                  edges will be removed.
      n...........the number of edges to remove.
    Returns:
      A *new* networkx Graph with n edges removed.
    In this doctest, we remove edges for two friends of D:
    >>> g = example_graph()
    >>> sorted(g.neighbors('D'))
    ['B', 'E', 'F', 'G']
    >>> train_graph = make_training_graph(g, 'D', 2)
    >>> sorted(train_graph.neighbors('D'))
    ['F', 'G']
    """

   
    g1 = graph.copy()
    count = 0
    a = sorted(g1.neighbors(test_node))
    count = 0
    for g in a:
        if count < n:
            #print(g)
            g1.remove_edge(test_node, g)
            count = count + 1
    return g1

def jaccard(graph, node, k):
    """
    Compute the k highest scoring edges to add to this node based on
    the Jaccard similarity measure.
    Note that we don't return scores for edges that already appear in the graph.
    Params:
      graph....a networkx graph
      node.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.
    Returns:
      A list of tuples in descending order of score representing the
      recommended new edges. Ties are broken by
      alphabetical order of the terminal node in the edge.
    In this example below, we remove edges (D, B) and (D, E) from the
    example graph. The top two edges to add according to Jaccard are
    (D, E), with score 0.5, and (D, A), with score 0. (Note that all the
    other remaining edges have score 0, but 'A' is first alphabetically.)
    >>> g = example_graph()
    >>> train_graph = make_training_graph(g, 'D', 2)
    >>> jaccard(train_graph, 'D', 2)
    [(('D', 'E'), 0.5), (('D', 'A'), 0.0)]
    """
    neighbors = set(graph.neighbors(node))
    scores = []
    for n in sorted(graph.nodes(),reverse = True):
        if n != node:
            if not graph.has_edge(node,n):
            
                neighbors2 = set(graph.neighbors(n))
                scores.append(((node,n), 1. * len(neighbors & neighbors2) / len(neighbors | neighbors2)))
        
    return sorted(scores, key=lambda t: (-t[-1],t[0]))[:k]



def path_score(graph, root, k, beta):
    """
    Compute a new link prediction scoring function based on the shortest
    paths between two nodes, as defined above.
    Note that we don't return scores for edges that already appear in the graph.
    This algorithm should have the same time complexity as bfs above.
    Params:
      graph....a networkx graph
      root.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.
      beta.....the beta parameter in the equation above.
    Returns:
      A list of tuples in descending order of score. Ties are broken by
      alphabetical order of the terminal node in the edge.
    In this example below, we remove edge (D, F) from the
    example graph. The top two edges to add according to path_score are
    (D, F), with score 0.5, and (D, A), with score .25. (Note that (D, C)
    is tied with a score of .25, but (D, A) is first alphabetically.)
    >>> g = example_graph()
    >>> train_graph = g.copy()
    >>> train_graph.remove_edge(*('D', 'F'))
    >>> path_score(train_graph, 'D', k=4, beta=.5)
    [(('D', 'F'), 0.5), (('D', 'A'), 0.25), (('D', 'C'), 0.25)]
    """
    path_score = []
    node2distances, node2num_paths, node2parents=bfs(graph, root, 2)
    for node in graph.nodes():
        if node != root:
            if not graph.has_edge(root,node):
                a = node2distances[node]
                b = node2num_paths[node]
                path_score.append((node, ((beta) ** a)* b ))
    return sorted(path_score, key=lambda x: x[1], reverse=True)


def evaluate(predicted_edges, graph):
    """
    Return the fraction of the predicted edges that exist in the graph.
    Args:
      predicted_edges...a list of edges (tuples) that are predicted to
                        exist in this graph
      graph.............a networkx Graph
    Returns:
      The fraction of edges in predicted_edges that exist in the graph.
    In this doctest, the edge ('D', 'E') appears in the example_graph,
    but ('D', 'A') does not, so 1/2 = 0.5
    >>> evaluate([('D', 'E'), ('D', 'A')], example_graph())
    0.5
    """
    evaluate = []
    for edge in predicted_edges:
        credit_dict[edge]=1
        evaluate = (credit_dict[edge]/len(predicted_edges))
    return evaluate
        

def main():
    graph= read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))      
        
    node2distances,node2num_paths, node2parents=bfs(graph, 'E', 5)
    #print(node2distances)
    #print(node2num_paths)
    #print(node2parents)
    bottom_up('E', node2distances,node2num_paths,node2parents)
    approximate_betweenness(graph, 2)
    partition_girvan_newman(graph, 5)
    #score_max_depths(graph, 5)
    make_training_graph(graph, 'D', 2)
    jaccard(graph, 'D', 2)
    path_score(graph, 'D', 4, 0.5)
    evaluate(predicted_edges, graph)
    '''for k,v in  s:
        v = len(s[1])
    print(s)'''
    

if __name__ == '__main__':
    main()
