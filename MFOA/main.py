import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import networkx as nx
from operator import itemgetter
from abc import ABC
import time

def v1(bestSmell, fly, V_r=0.4):
    items = floor(V_r * bestSmell.size)
    replace = bestSmell[:items]
    i = 0
    newfly = fly.copy()
    for val in fly:
        if val in replace:
            newfly[np.where(fly == val)[0][0]] = replace[i]
            i += 1
            if i == items:
                break
    return newfly

def v3(bestSmell, fly, V_r=0.4):
    items = floor(V_r * bestSmell.size)
    start = np.random.randint(0, len(fly)-1)
    bestSmell = np.append(bestSmell, bestSmell)
    replace = bestSmell[start:start+items]
    
    newfly = np.array([],dtype=int)
    for val in fly:
        if not val in replace:
            newfly = np.append(newfly,[val])
        if val in replace and not val in newfly:
            newfly = np.append(newfly, replace)
            
    return newfly

def mutate(fly):
    i1 = np.random.randint(0, len(fly)-1)
    while True:
        i2 = np.random.randint(0, len(fly)-1)
        if i1 != i2:
            break
    newfly = fly.copy()
    newfly[i1], newfly[i2] = fly[i2], fly[i1]
    return newfly

def s1(problem, fly, NN=10):
    best_fitness = problem.eval_fitness(fly)
    best_fly = fly
    for _ in range(NN):
        newfly = mutate(fly)
        new_fitness = problem.eval_fitness(newfly)
        if new_fitness > best_fitness:
            best_fitness = new_fitness
            best_fly = newfly
    return best_fly

def foa(problem, pop_size=200, max_attempts=10,
                max_iters=2500, curve=False, random_state=None, V_r=0.4, NN=10):
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
                
            best_fly = s1(problem, fly, NN)
            next_gen.append(best_fly)
        if not is_smell_concentration_equal:
            next_gen = np.array(next_gen)
            problem.set_population(next_gen)
            next_gen = []
            currentBest = problem.best_child()
            
            for i in range(pop_size):
                # Select parents
                fly = problem.get_population()[i]
                if not np.array_equal(fly, currentBest):
                    fly = v3(currentBest, fly, V_r)
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

class TravellingSalesDirected(mlrose.fitness.TravellingSales):
    """Fitness function for Travelling Salesman optimization problem.
    Evaluates the fitness of a tour of n nodes, represented by state vector
    :math:`x`, giving the order in which the nodes are visited, as the total
    distance travelled on the tour (including the distance travelled between
    the final node in the state vector and the first node in the state vector
    during the return leg of the tour). Each node must be visited exactly
    once for a tour to be considered valid.

    Parameters
    ----------
    coords: list of pairs, default: None
        Ordered list of the (x, y) coordinates of all nodes (where element i
        gives the coordinates of node i). This assumes that travel between
        all pairs of nodes is possible. If this is not the case, then use
        :code:`distances` instead.

    distances: list of triples, default: None
        List giving the distances, d, between all pairs of nodes, u and v, for
        which travel is possible, with each list item in the form (u, v, d).
        Order of the nodes does not matter, so (u, v, d) and (v, u, d) are
        considered to be the same. If a pair is missing from the list, it is
        assumed that travel between the two nodes is not possible. This
        argument is ignored if coords is not :code:`None`.

    Examples
    --------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose
        >>> import numpy as np
        >>> coords = [(0, 0), (3, 0), (3, 2), (2, 4), (1, 3)]
        >>> dists = [(0, 1, 3), (0, 2, 5), (0, 3, 1), (0, 4, 7), (1, 3, 6),
                     (4, 1, 9), (2, 3, 8), (2, 4, 2), (3, 2, 8), (3, 4, 4)]
        >>> fitness_coords = mlrose.TravellingSales(coords=coords)
        >>> state = np.array([0, 1, 4, 3, 2])
        >>> fitness_coords.evaluate(state)
        13.86138...
        >>> fitness_dists = mlrose.TravellingSales(distances=dists)
        >>> fitness_dists.evaluate(state)
        29

    Note
    ----
    1. The TravellingSales fitness function is suitable for use in travelling
       salesperson (tsp) optimization problems *only*.
    2. It is necessary to specify at least one of :code:`coords` and
       :code:`distances` in initializing a TravellingSales fitness function
       object.
    """

    def __init__(self, coords=None, distances=None):
        super().__init__(coords, distances)

        # Split into separate lists
        node1_list, node2_list, dist_list = zip(*distances)

        if min(dist_list) < 0:
            raise Exception("""The distance between each pair of nodes"""
                            + """ must be greater than or equal to 0.""")
        if min(node1_list + node2_list) < 0:
            raise Exception("""The minimum node value must be 0.""")

        if not max(node1_list + node2_list) == \
                (len(set(node1_list + node2_list)) - 1):
            raise Exception("""All nodes must appear at least once in"""
                            + """ distances.""")

        path_list = list(zip(node1_list, node2_list))

        self.distances = distances
        self.path_list = path_list
        self.dist_list = dist_list

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation. Each integer between 0 and
            (len(state) - 1), inclusive must appear exactly once in the array.

        Returns
        -------
        fitness: float
            Value of fitness function. Returns :code:`np.inf` if travel between
            two consecutive nodes on the tour is not possible.
        """

        if self.is_coords and len(state) != len(self.coords):
            raise Exception("""state must have the same length as coords.""")

        if not len(state) == len(set(state)):
            raise Exception("""Each node must appear exactly once in state.""")

        if min(state) < 0:
            raise Exception("""All elements of state must be non-negative"""
                            + """ integers.""")

        if max(state) >= len(state):
            raise Exception("""All elements of state must be less than"""
                            + """ len(state).""")

        fitness = 0

        # Calculate length of each leg of journey
        for i in range(len(state) - 1):
            node1 = state[i]
            node2 = state[i + 1]

            if self.is_coords:
                fitness += np.linalg.norm(np.array(self.coords[node1])
                                          - np.array(self.coords[node2]))
            else:
                path = (node1, node2)
                
                if path in self.path_list:
                    fitness += self.dist_list[self.path_list.index(path)]
                else:
                    fitness += np.inf

        # Calculate length of final leg
        node1 = state[-1]
        node2 = state[0]

        if self.is_coords:
            fitness += np.linalg.norm(np.array(self.coords[node1])
                                      - np.array(self.coords[node2]))
        else:
            path = (node1, node2)

            if path in self.path_list:
                fitness += self.dist_list[self.path_list.index(path)]
            else:
                fitness += np.inf

        return fitness


class Warehouse(ABC):
    def __init__(self, nRows=10, lotsPerRow=10):
        self.G = nx.Graph()
        self.randomPositions = []
        self.nRows = nRows
        self.lotsPerRow = lotsPerRow
        self.paths = []
        self.iterations = 0

    def Block(self):
        board = np.zeros((3*self.nRows,self.lotsPerRow+2))
        for i in range(1,3*self.nRows,3):
            board[i-1,1:-1]=1
            board[i+1,1:-1]=1
        board[1,0]=1/4
        return board

    def addTargetsToBlock(self, board):
        for target in self.randomPositions:
            x = 1 + 3*floor(target[0]/2)+ (1 if target[0]%2 == 1 else -1)
            y = target[1]
            board[x,y] = 1/2

    def createGraph(self, nLots=5):
        '''creates a Graph with nRows rows of lotsPerRow lots each, and nLots random lots'''
        rng = np.random.RandomState(2)
        for _ in range(nLots):
            randomLocation = (rng.randint(0,2*self.nRows),rng.randint(1,self.lotsPerRow))
            self.randomPositions.append(randomLocation)

        nodePositions = [(floor(x/2),y) for x,y in self.randomPositions]
        nodePositions = sorted(nodePositions, key=itemgetter(0,1))

        self.G.add_node(0, pos=(0,0))
        for i in range(len(nodePositions)):
            self.G.add_node(i+1, pos=nodePositions[i])

        nNodes = len(nodePositions) + 1
        edges = []
        frontAisle = 0
        backAisle = 1
        j = 1
        last = 0
        for i in range(self.nRows):
            if i!=0:
                self.G.add_node(nNodes, pos=(i,0))
                edges.append((frontAisle, nNodes, {"weight": 3}))
                frontAisle = nNodes
                last = nNodes
                nNodes += 1
            y = 0
            while (j <= len(nodePositions) and nodePositions[j-1][0] == i):
                edges.append((last, j, {"weight": nodePositions[j-1][1]-y}))
                y = nodePositions[j-1][1]
                last = j
                j += 1

            self.G.add_node(nNodes, pos=(i,self.lotsPerRow+1))
            edges.append((last, nNodes, {"weight": self.lotsPerRow+1-y}))
            if i!=0:
                edges.append((backAisle, nNodes, {"weight": 3}))
            backAisle = nNodes
            nNodes += 1

        self.G.add_edges_from(edges)

    def findShortestPath(self, nLots):
        dist = []
        all_paths_iter = nx.all_pairs_dijkstra(self.G)
        # first node done -- now process the rest
        for u, (distance, path) in all_paths_iter:
            if u >= nLots:
                break
            for i in range(u+1,nLots+1):
                if i != u:
                    dist.append((u,i,distance[i]))
                    self.paths.append((u,i,path[i]))
        return dist

    def plotRoutes(self, route):
        board = np.zeros((3*self.nRows,self.lotsPerRow+2))
        pos=nx.get_node_attributes(self.G,'pos')
        for i in range(len(route)):
            p1 = pos[route[i-1]]
            p2 = pos[route[i]]
            if p1[0] == p2[0]:
                y1 = min(p1[1],p2[1])
                y2 = max(p1[1],p2[1])
                x = 1 + 3*p1[0]
                board[x,y1:y2+1] = 1/2
            else:
                x1 = 1 + 3*min(p1[0],p2[0])
                x2 = 1 + 3*max(p1[0],p2[0])
                y = p1[1]
                board[x1:x2+1,y] = 1/2
        return board

    def edgeListFromState(self, state):
        route = np.array([])
        for i in range(len(state)):
            for source, target, path in self.paths:
                if source == state[i] and target == state[(i+1)%len(state)]:
                    route = np.concatenate((route, path), axis=0)
                    break
                elif source == state[(i+1)%len(state)] and target == state[i]:
                    route = np.concatenate((route, path[::-1]), axis=0)
                    break

        return [(route[i],route[i+1]) for i in range(len(route)-1)] , route

    def init_problem(self, count=22):
        if self.iterations == 0:
            self.createGraph(nLots=count)
            self.mDist = self.findShortestPath(count)

        self.iterations += 1
        
        problem_fit = mlrose.TSPOpt(length = count+1, distances = self.mDist, maximize=False)
        return problem_fit
    
    def solve(self, count=22, pop_size=200, V_r=0.4, NN=10):
        problem_fit = self.init_problem(count)
        return foa(problem_fit, pop_size=pop_size, V_r=V_r, NN=NN)
    
    def solve_ga(self, count=22):
        problem_fit = self.init_problem(count)
        return mlrose.genetic_alg(problem_fit, mutation_prob = 0.2,
                                              max_attempts = 100)
    
    def solve_sa(self, count=22):
        problem_fit = self.init_problem(count)
        return mlrose.simulated_annealing(problem_fit)
    
    def sshape(self, count=22):
        if self.iterations == 0:
            self.createGraph(nLots=count)
            self.mDist = self.findShortestPath(count)

        self.iterations += 1

        pos=nx.get_node_attributes(self.G,'pos')

        dirUp = True
        best_state = []
        best_fitness = 0
        lotsPerRow = {}
        for i in range(count+1):
            if pos[i][0] not in lotsPerRow:
                lotsPerRow[pos[i][0]] = [i]
            else:
                lotsPerRow[pos[i][0]].append(i)

        last_row = 0
        for i in range(self.nRows):
            if i in lotsPerRow:
                best_fitness += (i - last_row) * 3
                last_row = i
                if dirUp:
                    best_state = best_state + lotsPerRow[i]
                else:
                    best_state = best_state + lotsPerRow[i][::-1]
                
                dirUp = not dirUp
                best_fitness += self.lotsPerRow + 1
        if not dirUp:
            best_fitness += self.lotsPerRow + 1
        best_fitness += last_row * 3
        
        return best_state, best_fitness

    def midpoint(self, count=22):
        if self.iterations == 0:
            self.createGraph(nLots=count)
            self.mDist = self.findShortestPath(count)

        self.iterations += 1

        pos=nx.get_node_attributes(self.G,'pos')

        best_state = []
        best_fitness = 0
        lotsPerRowFront = {}
        lotsPerRowBack = {}
        for i in range(count+1):
            key = pos[i][0]
            y = pos[i][1]
            if y > self.lotsPerRow/2 and key not in lotsPerRowBack:
                lotsPerRowBack[key] = [i]
            elif y <= self.lotsPerRow/2 and key not in lotsPerRowFront:
                lotsPerRowFront[key] = [i]
            elif y > self.lotsPerRow/2:
                lotsPerRowBack[key].append(i)
            else:
                lotsPerRowFront[key].append(i)

        last_row = 0
        mymax = max(lotsPerRowFront.keys())
        for i in range(self.nRows):
            if i in lotsPerRowFront:
                best_fitness += (i - last_row) * 3
                last_row = i
                best_state = best_state + lotsPerRowFront[i]
                
                if i == mymax:
                    best_fitness += self.lotsPerRow + 1
                else:
                    best_fitness += 2 * pos[lotsPerRowFront[i][-1]][1]

        mymin = min(lotsPerRowBack.keys())
        for i in range(self.nRows,0,-1):
            if i in lotsPerRowBack:
                best_fitness += abs(i - last_row) * 3
                last_row = i
                best_state = best_state + lotsPerRowBack[i]
                
                if i == mymin:
                    best_fitness += self.lotsPerRow + 1
                else:
                    best_fitness += 2 * pos[lotsPerRowBack[i][0]][1]
                
        best_fitness += last_row * 3
        
        return best_state, best_fitness
    
    def plot_problem(self):
        pos=nx.get_node_attributes(self.G,'pos')
        nx.draw(self.G, pos, with_labels=True, font_weight='bold')
        plt.show()

    def plot(self, best_state, both=False):
        myedgelist, route = self.edgeListFromState(best_state)
        newedgelist = []
        for val in myedgelist:
            if val[0] != val[1]:
                newedgelist.append(val)
        myedgelist = newedgelist
        B = self.Block()
        self.addTargetsToBlock(B)
        pos=nx.get_node_attributes(self.G,'pos')
        nx.draw(self.G.to_directed(), pos, edgelist=myedgelist, with_labels=True, font_weight='bold')
        if both:
            plt.figure()
            plt.imshow(B)
            plt.imshow(self.plotRoutes(route), alpha=0.5, cmap=plt.cm.gray)
        plt.show()

class WarehouseOneDirection(Warehouse):
    def __init__(self, nRows=10, lotsPerRow=10):
        super().__init__(nRows, lotsPerRow)
        self.G = nx.DiGraph()

    def createGraph(self, nLots=5):
        '''creates a Graph with nRows rows of lotsPerRow lots each, and nLots random lots'''
        rng = np.random.RandomState(2)
        for _ in range(nLots):
            randomLocation = (rng.randint(0,2*self.nRows),rng.randint(1,self.lotsPerRow))
            self.randomPositions.append(randomLocation)

        nodePositions = [(floor(x/2),y) for x,y in self.randomPositions]
        nodePositions = sorted(nodePositions, key=itemgetter(0,1))

        self.G.add_node(0, pos=(0,0))
        for i in range(len(nodePositions)):
            self.G.add_node(i+1, pos=nodePositions[i])

        nNodes = len(nodePositions) + 1
        edges = []
        frontAisle = 0
        backAisle = 1
        j = 1
        last = 0
        for i in range(self.nRows):
            if i!=0:
                self.G.add_node(nNodes, pos=(i,0))
                edges.append((nNodes, frontAisle, {"weight": 3}))
                frontAisle = nNodes
                last = nNodes
                nNodes += 1
            y = 0
            while (j <= len(nodePositions) and nodePositions[j-1][0] == i):
                if i%2 == 0:
                    edges.append((last, j, {"weight": nodePositions[j-1][1]-y}))
                if i%2 == 1:
                    edges.append((j, last, {"weight": nodePositions[j-1][1]-y}))
                y = nodePositions[j-1][1]
                last = j
                j += 1

            self.G.add_node(nNodes, pos=(i,self.lotsPerRow+1))
            if i%2 == 0:
                edges.append((last, nNodes, {"weight": self.lotsPerRow+1-y}))
            if i%2 == 1:
                edges.append((nNodes, last, {"weight": self.lotsPerRow+1-y}))
            
            if i!=0:
                edges.append((backAisle, nNodes, {"weight": 3}))
            backAisle = nNodes
            nNodes += 1

        self.G.add_edges_from(edges)

    def findShortestPath(self, nLots):
        dist = []
        all_paths_iter = nx.all_pairs_dijkstra(self.G)
        # first node done -- now process the rest
        for u, (distance, path) in all_paths_iter:
            if u > nLots:
                break
            for i in range(nLots+1):
                if i != u:
                    dist.append((u,i,distance[i]))
                    self.paths.append((u,i,path[i]))
        return dist
        
    def edgeListFromState(self, state):
        route = np.array([])
        for i in range(len(state)):
            for source, target, path in self.paths:
                if source == state[i] and target == state[(i+1)%len(state)]:
                    route = np.concatenate((route, path), axis=0)
                    break

        return [(route[i],route[i+1]) for i in range(len(route)-1)] , route

    def init_problem(self, count=22):
        if self.iterations == 0:
            self.createGraph(nLots=count)
            self.mDist = self.findShortestPath(count)

        self.iterations += 1
        
        problem_fit = mlrose.TSPOpt(length = count+1, distances = self.mDist,  maximize=False,
                                    fitness_fn = TravellingSalesDirected(distances=self.mDist))
        return problem_fit

class WarehouseWithAisles(Warehouse):
    def __init__(self, nRows=10, lotsPerRow=10,aisles=[5]):
        super().__init__(nRows, lotsPerRow)
        self.aisles = aisles
    
    def Block(self):
        board = np.zeros((3*self.nRows,self.lotsPerRow+2))
        for i in range(1,3*self.nRows,3):
            board[i-1,1:-1]=1
            board[i+1,1:-1]=1
            for j in self.aisles:
                board[i-1,j]=0
            board[i+1,j]=0
        board[1,0]=1/4
        return board

    def createGraph(self, nLots=5):
        '''creates a Graph with nRows rows of lotsPerRow lots each, and nLots random lots'''
        rng = np.random.RandomState(2)
        for _ in range(nLots):
            randomLocation = (rng.randint(0,2*self.nRows),rng.randint(1,self.lotsPerRow-len(self.aisles)))
            self.randomPositions.append(randomLocation)

        nodePositions = [(floor(x/2),y) for x,y in self.randomPositions]
        nodePositions = sorted(nodePositions, key=itemgetter(0,1))

        self.G.add_node(0, pos=(0,0))
        for i in range(len(nodePositions)):
            self.G.add_node(i+1, pos=nodePositions[i])

        nNodes = len(nodePositions) + 1
        edges = []
        frontAisle = 0
        backAisle = 1
        j = 1
        last = 0
        lastAisleNodes = {}
        for i in range(self.nRows):
            if i!=0:
                self.G.add_node(nNodes, pos=(i,0))
                edges.append((frontAisle, nNodes, {"weight": 3}))
                frontAisle = nNodes
                last = nNodes
                nNodes += 1
            y = 0
            aisleIndex = 0
            while (j <= len(nodePositions) and nodePositions[j-1][0] == i):
                while aisleIndex < len(self.aisles) and nodePositions[j-1][1] > self.aisles[aisleIndex]:
                    self.G.add_node(nNodes, pos=(i,self.aisles[aisleIndex]))
                    edges.append((last, nNodes, {"weight": self.aisles[aisleIndex]-y}))
                    last = nNodes
                    if i!=0:
                        edges.append((lastAisleNodes[aisleIndex], nNodes, {"weight": 3}))
                    lastAisleNodes[aisleIndex] = nNodes
                    nNodes += 1
                    y = self.aisles[aisleIndex]
                    aisleIndex += 1
                    
                edges.append((last, j, {"weight": nodePositions[j-1][1]-y}))
                y = nodePositions[j-1][1]
                last = j
                j += 1

            while aisleIndex < len(self.aisles):
                self.G.add_node(nNodes, pos=(i,self.aisles[aisleIndex]))
                if i!=0:
                    edges.append((lastAisleNodes[aisleIndex], nNodes, {"weight": 3}))
                lastAisleNodes[aisleIndex] = nNodes
                edges.append((last, nNodes, {"weight": self.aisles[aisleIndex]-y}))
                last = nNodes
                nNodes += 1
                y = self.aisles[aisleIndex]
                aisleIndex += 1
            aisleIndex = 0

            self.G.add_node(nNodes, pos=(i,self.lotsPerRow+1))
            edges.append((last, nNodes, {"weight": self.lotsPerRow+1-y}))
            if i!=0:
                edges.append((backAisle, nNodes, {"weight": 3}))
            backAisle = nNodes
            nNodes += 1

        self.G.add_edges_from(edges)

class Rack(ABC):
    def __init__(self, nRows=10, lotsPerRow=10):
        self.G = nx.Graph()
        self.randomPositions = []
        self.nRows = nRows
        self.lotsPerRow = lotsPerRow
        self.iterations = 0

    def createGraph(self, nLots=5):
        '''creates a Graph with nRows rows of lotsPerRow lots each, and nLots random lots'''    
        nNodes = 1
        self.G.add_node(0, pos=(0,0))
        self.randomPositions.append((0,0))
        rng = np.random.RandomState(2)
        for _ in range(nLots):
            randomLocation = (rng.randint(0,self.nRows),rng.randint(0,self.lotsPerRow))
            self.randomPositions.append(randomLocation)
            self.G.add_node(nNodes, pos=randomLocation)
            nNodes += 1

    def edgeListFromState(self, state):
        return [(state[i],state[(i+1)%len(state)]) for i in range(len(state))] , state

    def init_problem(self, count=22):
        if self.iterations == 0:
            self.createGraph(nLots=count)

        self.iterations += 1
        
        problem_fit = mlrose.TSPOpt(length = count+1, coords= self.randomPositions, maximize=False)
        return problem_fit
    
    def solve(self, count=22):
        problem_fit = self.init_problem(count)
        return foa(problem_fit, pop_size=200, V_r=0.4, NN=10)
    
    def solve_ga(self, count=22):
        problem_fit = self.init_problem(count)
        return mlrose.genetic_alg(problem_fit, mutation_prob = 0.2,
                                              max_attempts = 100)
    
    def solve_sa(self, count=22):
        problem_fit = self.init_problem(count)
        return mlrose.simulated_annealing(problem_fit)
    
    def plot(self, best_state):
        myedgelist, _ = self.edgeListFromState(best_state)
        
        pos=nx.get_node_attributes(self.G,'pos')
        
        nx.draw(self.G, pos, with_labels=True, font_weight='bold')
        plt.figure()
        nx.draw(self.G.to_directed(), pos, edgelist=myedgelist, with_labels=True, font_weight='bold')
        plt.show()

results = {}
def simulate(fn, n, debug=False, name=None, **kwargs):
    """Simulate a function call n times and return the average time."""
    times = []
    fitnesses = []
    best_state = []
    best_fitness = np.inf
    global results
    if name == None:
        name = fn.__name__

    for _ in range(n):
        start = time.time()
        state, fitness = fn(**kwargs)
        if debug:
            print('The best state found is: ', state)
            print('The fitness at the best state is: ', fitness)
        end = time.time()
        times.append(end - start)
        fitnesses.append(fitness)
        if fitness < best_fitness:
            best_state = state
            best_fitness = fitness

    print("Fitness: ", np.mean(fitnesses), "+/-", np.std(fitnesses))
    print("Time: ", np.mean(times), "+/-", np.std(times))
    results[name] = [np.mean(fitnesses), np.std(fitnesses), np.mean(times), np.std(times)]
    return best_state, best_fitness

TEST_CASES = ["RACK","WAREHOUSE","WAREHOUSE_WITH_AISLES","WAREHOUSE_ONE_DIRECTION","HYPERPARAMETER_TUNING"]

def plotStatistics(rows):
    global results
    x = rows
    y = [results[entry][0] for entry in rows]
    err = [results[entry][1] for entry in rows]
    times = [results[entry][2] for entry in rows]
    timeerr = [results[entry][3] for entry in rows]
    plt.subplot(121)
    
    # Plot scatter here
    plt.bar(x, y)
    plt.errorbar(x, y, yerr=err, fmt="o", color="r")

    plt.subplot(122)

    plt.bar(x, times)
    plt.errorbar(x, times, yerr=timeerr, fmt="o", color="r")
    plt.show()

if __name__ == "__main__":
    testCase = "WAREHOUSE_ONE_DIRECTION"

    if testCase == "RACK":
        warehouse = Rack(10,30)
        count = 10
    elif testCase == "WAREHOUSE":
        warehouse = Warehouse(20,50)
        count = 20
    elif testCase == "WAREHOUSE_WITH_AISLES":
        warehouse = WarehouseWithAisles(10,30, aisles=[10,20])
        count = 20
    elif testCase == "WAREHOUSE_ONE_DIRECTION":
        warehouse = WarehouseOneDirection(20,50)
        count = 20
    elif testCase == "HYPERPARAMETER_TUNING":
        warehouse = Warehouse(20,50)
        count = 20

    functions = [warehouse.solve, warehouse.solve_ga, warehouse.solve_sa]

    '''
    for fn in [warehouse.midpoint, warehouse.sshape]:
        print(fn.__name__)
        best_state, best_fitness = simulate(fn, 1, count=count)

        print("Best fitness: ", best_fitness)
        #warehouse.plot(best_state)
    '''
    if not testCase == "HYPERPARAMETER_TUNING":
        for fn in functions:
            print(fn.__name__)
            
            best_state, best_fitness = simulate(fn, 5, count=count)

            print("Best fitness: ", best_fitness)
            warehouse.plot(best_state)

        ### Plotting
        x = ['FOA', 'GA', 'SA']
        y = [results['solve'][0], results['solve_ga'][0], results['solve_sa'][0]]
        err = [results['solve'][1], results['solve_ga'][1], results['solve_sa'][1]]

        # Plot scatter here
        plt.bar(x, y)
        plt.errorbar(x, y, yerr=err, fmt="o", color="r")

        plt.figure()

        times = [results['solve'][2], results['solve_ga'][2], results['solve_sa'][2]]
        timeerr = [results['solve'][3], results['solve_ga'][3], results['solve_sa'][3]]

        plt.bar(x, times)
        plt.errorbar(x, times, yerr=timeerr, fmt="o", color="r")
        plt.show()
    else:
        for pop_size in [100, 300]:
            print("Pop size: ", pop_size)
            best_state, best_fitness = simulate(warehouse.solve, 5, name=str(pop_size), count=count, pop_size=pop_size)

            print("Best fitness: ", best_fitness)

        plotStatistics(['100', '300'])
        
        for V_r in [0.1, 0.5]:
            print("V_r: ", V_r)
            best_state, best_fitness = simulate(warehouse.solve, 5, name=str(V_r), count=count, V_r=V_r)

            print("Best fitness: ", best_fitness)
        
        plotStatistics(['0.1', '0.5'])
        
        for NN in [5, 15]:
            print("NN: ", NN)
            best_state, best_fitness = simulate(warehouse.solve, 5, name=str(NN), count=count, NN=NN)

            print("Best fitness: ", best_fitness)
        
        plotStatistics(['5', '15'])
        
    