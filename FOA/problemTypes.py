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
from fitnessFunctions import *
from algorithms import FOA, EFOA

class AbstractWarehouse(ABC):
    def __init__(self, nRows=10, lotsPerRow=10):
        self.G = nx.Graph()
        self.randomPositions = []
        self.nRows = nRows
        self.lotsPerRow = lotsPerRow
        self.iterations = 0

    def createGraph(self, nLots=5):
        '''creates a Graph with nRows rows of lotsPerRow lots each, and nLots random lots'''    
        raise NotImplementedError

    def edgeListFromState(self, state):
        raise NotImplementedError

    def init_problem(self, count=22):
        raise NotImplementedError
    
    def solve(self, count=22, pop_size=200, V_r=0.4, NN=10, visionFn=None):
        problem_fit = self.init_problem(count)
        return FOA.foa(problem_fit, pop_size=pop_size, V_r=V_r, NN=NN, visionFn=visionFn)
    
    def solve_efoa(self, count=22, pop_size=200, curve=False):
        problem_fit = self.init_problem(count)
        return EFOA.efoa(problem_fit, pop_size=pop_size, curve=curve)
    
    def solve_ga(self, count=22):
        problem_fit = self.init_problem(count)
        return mlrose.genetic_alg(problem_fit, mutation_prob = 0.2,
                                              max_attempts = 100)
    
    def solve_sa(self, count=22):
        problem_fit = self.init_problem(count)
        return mlrose.simulated_annealing(problem_fit)
    
    def plot(self, best_state):
        raise NotImplementedError

class Warehouse(AbstractWarehouse):
    def __init__(self, nRows=10, lotsPerRow=10):
        super().__init__(nRows, lotsPerRow)
        self.paths = []

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
    
    def plot_problem(self, both=False):
        pos=nx.get_node_attributes(self.G,'pos')
        nx.draw(self.G, pos, with_labels=True, font_weight='bold')
        if both:
            B = self.Block()
            self.addTargetsToBlock(B)
            plt.figure()
            plt.imshow(B)
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
            while True:
                randomLocation = (rng.randint(0,2*self.nRows),rng.randint(1,self.lotsPerRow-len(self.aisles)))
                if randomLocation[1] not in self.aisles:
                    break
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

class Rack(AbstractWarehouse):
    def __init__(self, nRows=10, lotsPerRow=10):
        super().__init__(nRows, lotsPerRow)

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
    
    def plot(self, best_state):
        myedgelist, _ = self.edgeListFromState(best_state)
        
        pos=nx.get_node_attributes(self.G,'pos')
        
        nx.draw(self.G, pos, with_labels=True, font_weight='bold')
        plt.figure()
        nx.draw(self.G.to_directed(), pos, edgelist=myedgelist, with_labels=True, font_weight='bold')
        plt.show()

if __name__ == "__main__":
    warehouse = Rack(10,30)
    _,_,curve = warehouse.solve_efoa(10,curve=True)
    plt.plot(curve)
    plt.show()
    #best_state, _ = warehouse.solve(20)
    #warehouse.plot(best_state,both=True)

    """
    x = np.array([0.01*i for i in range(1,501)])
    c = 1
    mu = 0
    y = np.array([np.sqrt(c/2/np.pi)*np.exp(-c/(2*(i-mu)))/(i-mu)**(3/2) for i in x])

    plt.plot(x,y)
    plt.show()"""