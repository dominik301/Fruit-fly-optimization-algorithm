from calendar import c
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import networkx as nx
from itertools import chain

def v1(bestSmell, fly):
    V_r = 0.4
    items = floor(V_r * bestSmell.size)
    replace = bestSmell[:items]
    i = 0
    for val in fly:
        if val in replace:
            fly[np.where(fly == val)[0][0]] = replace[i]
            i += 1
            if i == items:
                break

def mutate(fly):
    i1 = np.random.randint(0, len(fly)-1)
    while True:
        i2 = np.random.randint(0, len(fly)-1)
        if i1 != i2:
            break
    newfly = fly.copy()
    newfly[i1], newfly[i2] = fly[i2], fly[i1]
    return newfly

def s1(problem, fly):
    N =10
    best_fitness = problem.eval_fitness(fly)
    best_fly = fly
    for _ in range(N):
        newfly = mutate(fly)
        new_fitness = problem.eval_fitness(newfly)
        if new_fitness > best_fitness:
            best_fitness = new_fitness
            best_fly = newfly
    return best_fly

def foa(problem, pop_size=200, max_attempts=10,
                max_iters=2500, curve=False, random_state=None):
    """Use a standard genetic algorithm to find the optimum for a given
    optimization problem.
    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    pop_size: int, default: 200
        Size of population to be used in genetic algorithm.
    mutation_prob: float, default: 0.1
        Probability of a mutation at each element of the state vector
        during reproduction, expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.
    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array of arrays containing the fitness of the entire population
        at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.
    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    if curve:
        fitness_curve = []

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0

    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Create next generation of population
        next_gen = []

        is_smell_concentration_equal = True

        currentBest = problem.best_child()
        currentFitness = problem.eval_fitness(currentBest)

        for i in range(pop_size):
            # Select parents
            fly = problem.get_population()[i]
            if currentFitness != problem.eval_fitness(fly):
                is_smell_concentration_equal = False
                break

        if (is_smell_concentration_equal):
            for i in range(pop_size):
                # Select parents
                fly = problem.get_population()[i]
                best_fly = s1(problem, fly)
                next_gen.append(best_fly)
        else:
            for i in range(pop_size):
                # Select parents
                fly = problem.get_population()[i]
                if not np.array_equal(fly, currentBest):
                    v1(currentBest, fly)
                next_gen.append(fly)

        next_gen = np.array(next_gen)
        problem.set_population(next_gen)

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0

        else:
            attempts += 1

        if curve:
            fitness_curve.append(problem.get_fitness())

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness

"""
# Create list of distances between pairs of cities
dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426), \
             (0, 5, 5.3852), (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000), \
             (1, 3, 2.8284), (1, 4, 2.0000), (1, 5, 4.1231), (1, 6, 4.2426), \
             (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), (2, 5, 4.4721), \
             (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056), \
             (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623), \
             (4, 7, 2.2361), (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]

# Initialize fitness function object using dist_list
fitness_dists = mlrose.TravellingSales(distances = dist_list)

problem_fit = mlrose.TSPOpt(length = 8, distances = dist_list, # fitness_fn = fitness_dists,
                            maximize=False)

best_state, best_fitness = foa(problem_fit, random_state = 2)

print('The best state found is: ', best_state)

print('The fitness at the best state is: ', best_fitness)
"""
def Checkerboard(N,n):
    """N: size of board; n=size of each square; N/(2*n) must be an integer """
    if (N%(2*n)):
        print('Error: N/(2*n) must be an integer')
        return False
    a = np.concatenate((np.zeros(n),np.ones(n)))
    print(a)
    b=np.pad(a,int((N**2)/2-n),'wrap').reshape((N,N))
    print(b)
    return (b+b.T==1).astype(int)

def Rows(N):
    """N: size of board"""
    board = np.zeros((N,N))
    for i in range(2,N-1,3):
        board[i-1:i+1,1:N-1]=1
    return board

def targets(count, N, board):
    target = []
    for i in range(count):
        y = np.random.randint(1,N-1)
        valid = False
        while not valid:
            x = np.random.randint(1,N-1)
            if (board[x,y] != 0):
                valid = True
        target.append((x,y))
        board[x,y] = 1/2
    return target

def get_next(x):
    if x % 3 == 1:
        return x-1
    else:
        return x+1

def distances(target, N):
    dist = []
    for i in range(len(target)):
        for j in range(i+1,len(target)):
            x11 = target[i][1]+target[j][1]
            x12 = 2 * (N-1) - target[i][1] - target[j][1]
            x2 = abs(get_next(target[i][0])-get_next(target[j][0]))
            if x2 == 0:
                x11 = abs(target[i][1]-target[j][1])
            mDist = max(x2 + min(x11,x12), 0.01)

            dist.append((i,j,mDist))
    return dist

@nx.utils.not_implemented_for('directed')
def metric_closure(G, weight='weight'):
    """  Return the metric closure of a graph.
    The metric closure of a graph *G* is the complete graph in which each edge
    is weighted by the shortest path distance between the nodes in *G* .
    Parameters
    ----------
    G : NetworkX graph
    Returns
    -------
    NetworkX graph
        Metric closure of the graph `G`.
    """
    M = nx.Graph()

    Gnodes = set(G)

    # check for connected graph while processing first node
    all_paths_iter = nx.all_pairs_dijkstra(G, weight=weight)
    u, (distance, path) = next(all_paths_iter)
    if Gnodes - set(distance):
        msg = "G is not a connected graph. metric_closure is not defined."
        raise nx.NetworkXError(msg)
    Gnodes.remove(u)
    for v in Gnodes:
        M.add_edge(u, v, distance=distance[v], path=path[v])

    # first node done -- now process the rest
    for u, (distance, path) in all_paths_iter:
        Gnodes.remove(u)
        for v in Gnodes:
            M.add_edge(u, v, distance=distance[v], path=path[v])

    return M

@nx.utils.not_implemented_for('multigraph')
@nx.utils.not_implemented_for('directed')
def steiner_tree(G, terminal_nodes, weight='weight'):
    """ Return an approximation to the minimum Steiner tree of a graph.
    Parameters
    ----------
    G : NetworkX graph
    terminal_nodes : list
         A list of terminal nodes for which minimum steiner tree is
         to be found.
    Returns
    -------
    NetworkX graph
        Approximation to the minimum steiner tree of `G` induced by
        `terminal_nodes` .
    Notes
    -----
    Steiner tree can be approximated by computing the minimum spanning
    tree of the subgraph of the metric closure of the graph induced by the
    terminal nodes, where the metric closure of *G* is the complete graph in
    which each edge is weighted by the shortest path distance between the
    nodes in *G* .
    This algorithm produces a tree whose weight is within a (2 - (2 / t))
    factor of the weight of the optimal Steiner tree where *t* is number of
    terminal nodes.
    """
    # M is the subgraph of the metric closure induced by the terminal nodes of
    # G.
    M = metric_closure(G, weight=weight)
    # Use the 'distance' attribute of each edge provided by the metric closure
    # graph.
    H = M.subgraph(terminal_nodes)
    mst_edges = nx.minimum_spanning_edges(H, weight='distance', data=True)
    # Create an iterator over each edge in each shortest path; repeats are okay
    edges = chain.from_iterable(nx.utils.pairwise(d['path']) for u, v, d in mst_edges)
    T = G.edge_subgraph(edges)
    return T


def adjacency_matrix_to_graph(adjacency_matrix):
    node_weights = [adjacency_matrix[i][i] for i in range(len(adjacency_matrix))]
    adjacency_matrix_formatted = [[0 if entry == 'x' else entry for entry in row] for row in adjacency_matrix]

    for i in range(len(adjacency_matrix_formatted)):
        adjacency_matrix_formatted[i][i] = 0

    G = nx.convert_matrix.from_numpy_array(np.matrix(adjacency_matrix_formatted))

    message = ''

    for node, datadict in G.nodes.items():
        if node_weights[node] != 'x':
            message += 'The location {} has a road to itself. This is not allowed.\n'.format(node)
        datadict['weight'] = node_weights[node]

    return G, message

def convert_locations_to_indices(list_to_convert, list_of_locations):
    return [list_of_locations.index(name) if name in list_of_locations else None for name in list_to_convert]

def customDistance(adjacency_matrix, list_of_locations, list_of_homes, starting_car_location):
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    list_of_homes_index = convert_locations_to_indices(list_of_homes, list_of_locations)
    startingCarIndex = convert_locations_to_indices([starting_car_location], list_of_locations)[0]
    steinerTree = steiner_tree(G, list_of_homes_index + [startingCarIndex])
    #print("Steiner Tree")
    treeTraversal = list(nx.algorithms.traversal.dfs_preorder_nodes(steinerTree, source=startingCarIndex))
    steinerCarCycle = []
    for i in range(len(treeTraversal)-1):
        start = treeTraversal[i]
        end = treeTraversal[i+1]
        #SteinerTree
        shortest_path = nx.shortest_path(steinerTree, start, end)
        steinerCarCycle.extend(shortest_path[:len(shortest_path)-1])
    #SteinerTree
    lastShortPath = nx.shortest_path(steinerTree, treeTraversal[len(treeTraversal)-1], startingCarIndex)
    steinerCarCycle.extend(lastShortPath)
    print(G)
    print(steinerTree)
    print(steinerCarCycle)
    print(treeTraversal)

def plotRoutes(best_state, mTargets):
    board = np.zeros((N,N))
    for i in range(len(best_state)-1):
        x1 = get_next(mTargets[best_state[i]][0])
        y1 = mTargets[best_state[i]][1]
        x2 = get_next(mTargets[best_state[i+1]][0])
        y2 = mTargets[best_state[i+1]][1]
        if x1 == x2:
            board[x1,min(y1,y2):max(y1,y2)+1] += 1
        else:
            y11 = y1+y2
            y12 = 2 * (N-1) - y1 - y2
            if abs(y11) < abs(y12):
                board[x1,:y1+1] += 1
                board[min(x1,x2)+1:max(x1,x2),0] += 1
                board[x2,:y2+1] += 1
            else:
                board[x1,y1:] += 1
                board[min(x1,x2)+1:max(x1,x2),-1] += 1
                board[x2,y2:] += 1
    for entry in range(len(board)):
        board[entry] = (pow(2,board[entry]) - 1)/ pow(2,board[entry])
    return board
    
"""
#B=Checkerboard(10,1)
N = 100
count = 22
B = Rows(N)
mTargets = targets(count,N,B)
mDist = distances(mTargets,N)

fitness_dists = mlrose.TravellingSales(distances = mDist)

problem_fit = mlrose.TSPOpt(length = count, distances = mDist, # fitness_fn = fitness_dists,
                            maximize=False)

best_state, best_fitness = foa(problem_fit, random_state = 2, max_attempts = 20)

print('The best state found is: ', best_state)

for val in best_state:
    print(val, mTargets[val])

print('The fitness at the best state is: ', best_fitness)

plt.imshow(B)

plt.imshow(plotRoutes(best_state, mTargets), alpha=0.5, cmap=plt.cm.gray)
plt.show()
"""


list_of_locations = [i for i in range(1,51)]
list_of_houses = [28,31,49,42,34,14,47,45,19,9,40,17,30,21,20,24,1,32,25,4,12,46,48,36,22]
starting_location = 22
adjacency_matrix = [[0]*50]*50
for i in range(50):
    for j in range(50):
        adjacency_matrix[i][j] = 'x'
adjacency_matrix[0][10] = 0.434
adjacency_matrix[0][21] = 0.802
adjacency_matrix[0][22] = 0.372
adjacency_matrix[0][31] = 0.36
adjacency_matrix[5][1] = 0.364
adjacency_matrix[8][1] = 0.33
adjacency_matrix[10][1] = 0.434
adjacency_matrix[46][1] = 0.888
customDistance(adjacency_matrix,list_of_locations,list_of_houses, starting_location)
#adjacency_matrix = [[entry if entry == 'x' else float(entry) for entry in row] for row in input_data[5:]]