import numpy as np
from math import floor
import networkx as nx
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import product, permutations

class FOA:
    def v1(bestSmell, fly, V_r=0.4, problem=None):
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

    def v3(bestSmell, fly, V_r=0.4, problem=None):
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
            newfly = FOA.mutate(fly)
            new_fitness = problem.eval_fitness(newfly)
            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_fly = newfly
        return best_fly

    def foa(problem, smellFn=None, visionFn=None, pop_size=200, max_attempts=10,
                    max_iters=2500, curve=False, random_state=None, V_r=0.4, NN=10):
        """Use Fruit Fly Optimization Algorithm to find the optimum for a given
        optimization problem.
        Parameters
        ----------
        problem: optimization object
            Object containing fitness function optimization problem to be solved.
            For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
            :code:`TSPOpt()`.
        pop_size: int, default: 200
            Size of population to be used in genetic algorithm.
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
            plt.axis()

        if smellFn is None:
            smellFn = FOA.s1
        if visionFn is None:
            visionFn = FOA.v3

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
                    
                best_fly = smellFn(problem, fly, NN)
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
                        fly = visionFn(currentBest, fly, V_r, problem)
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
                plt.plot(fitness_curve)
                plt.pause(0.0001)

        best_fitness = problem.get_maximize()*problem.get_fitness()
        best_state = problem.get_state()

        if curve:
            return best_state, best_fitness, np.asarray(fitness_curve)

        return best_state, best_fitness
    
    def crossover(bestSmell, fly, V_r=0, problem=None):
        newfly = problem.reproduce(fly, bestSmell, 0)
        if problem.eval_fitness(newfly) > problem.eval_fitness(fly):
            fly = newfly     
        return fly
    
    def ifoa(problem, **kwargs):
        return FOA.foa(problem, visionFn=FOA.crossover, **kwargs)
        

class EFOA:
    def reverseOperatorCoords(fly:np.array, coords, G=None, debug=False):
        i = np.random.randint(0, len(fly))
        posList = coords.copy()
        posList.remove(coords[i])
        A = np.array(posList)
        j = KDTree(A).query(coords[i])[1]
        if j >= i:
            j += 1
        idx1 = np.where(fly==i)[0][0]
        idx2 = np.where(fly==j)[0][0]
        if debug:
            print("i: ", i)
            print("j: ", j)
            print("fly: ", fly)
            newfly = EFOA.TR(fly, idx1, idx2)
            pos=nx.get_node_attributes(G,'pos')
            ax1 = plt.subplot(121)
            myedgelist = [(fly[i],fly[(i+1)%len(fly)]) for i in range(len(fly))]
            nx.draw(G.to_directed(), pos, edgelist=myedgelist, with_labels=True, font_weight='bold')
            ax2 = plt.subplot(122)
            myedgelist = [(newfly[i],newfly[(i+1)%len(newfly)]) for i in range(len(newfly))]
            nx.draw(G.to_directed(), pos, edgelist=myedgelist, with_labels=True, font_weight='bold')
            ax1.title.set_text('F')
            ax2.title.set_text("F'")
            plt.show()
            
        return EFOA.TR(fly, idx1, idx2)
    
    def reverseOperatorDists(fly:np.array, dists):
        i = np.random.randint(0, len(fly))
        shortestDists = sorted(dists, key=itemgetter(2))
        for val in shortestDists:
            n1,n2,_ = val
            if n1 == i:
                j = n2
                break
            elif n2 == i:
                j = n1
                break
        idx1 = np.where(fly==i)[0][0]
        idx2 = np.where(fly==j)[0][0]
        return EFOA.TR(fly, idx1, idx2)

    def TR(fly, idx1, idx2):
        newfly = []
        if idx1 < idx2:
            if idx2 != len(fly)-1:
                newfly = np.concatenate((fly[:idx1+1], fly[idx2:idx1:-1], fly[idx2+1:]),axis=None)
            else:
                newfly = np.concatenate((fly[:idx1+1], fly[idx2:idx1:-1]),axis=None)
        else:
            if idx2!=0:
                newfly = np.concatenate((fly[:idx2], fly[idx1-1:idx2-1:-1], fly[idx1:]),axis=None)
            else:
                newfly = np.concatenate((fly[idx1-1::-1], fly[idx1:]),axis=None)
        return np.array(newfly)

    def plot(G, flyA, flyB, A1, A2, A3):
        pos=nx.get_node_attributes(G,'pos')
        ax1 = plt.subplot(231)
        myedgelist = [(flyA[i],flyA[(i+1)%len(flyA)]) for i in range(len(flyA))]
        nx.draw(G.to_directed(), pos, edgelist=myedgelist, with_labels=True, font_weight='bold')
        ax2 = plt.subplot(232)
        myedgelist = [(flyB[i],flyB[(i+1)%len(flyB)]) for i in range(len(flyB))]
        nx.draw(G.to_directed(), pos, edgelist=myedgelist, with_labels=True, font_weight='bold')
        ax4 = plt.subplot(234)
        myedgelist = [(A1[i],A1[(i+1)%len(A1)]) for i in range(len(A1))]
        nx.draw(G.to_directed(), pos, edgelist=myedgelist, with_labels=True, font_weight='bold')
        ax5 = plt.subplot(235)
        myedgelist = [(A2[i],A2[(i+1)%len(A2)]) for i in range(len(A2))]
        nx.draw(G.to_directed(), pos, edgelist=myedgelist, with_labels=True, font_weight='bold')
        ax6 = plt.subplot(236)
        myedgelist = [(A3[i],A3[(i+1)%len(A3)]) for i in range(len(A3))]
        nx.draw(G.to_directed(), pos, edgelist=myedgelist, with_labels=True, font_weight='bold')
        ax1.title.set_text('flyA')
        ax2.title.set_text('flyB')
        ax4.title.set_text('A1')
        ax5.title.set_text('A2')
        ax6.title.set_text('A3')
        plt.show()

    def multiplicationOperator(flyA:np.array, flyB:np.array, problem, G=None, debug=False):
        randIdx = np.random.randint(0, len(flyB)-1)
        bi = flyB[randIdx]
        bib = flyB[(randIdx-1) % len(flyB)]
        bia = flyB[(randIdx+1) % len(flyB)]
        j = np.where(flyA==bib)[0][0]
        k = np.where(flyA==bi)[0][0]
        m = np.where(flyA==bia)[0][0]
        A1 = EFOA.TR(flyA, k, j)
        A2 = EFOA.TR(flyA, k, m)
        A3 = EFOA.TR(A1, k, m)
        f1 = problem.eval_fitness(A1)
        f2 = problem.eval_fitness(A2)
        f3 = problem.eval_fitness(A3)
        if debug:
            print("A", flyA)
            print("B", flyB)
            print("bi", bi)
            print("A1", A1)
            print("A2", A2)
            print("A3", A3)
            EFOA.plot(G, flyA, flyB, A1, A2, A3)
            print("Fitnesses", f1, f2, f3)
        best = np.argmax([f1, f2, f3])
        solutions = [A1, A2, A3]
        return solutions[best]

    def efoa(problem, pop_size=200, max_attempts=10,
                    max_iters=2500, p=0.1, curve=False, random_state=None):
        """Use Elimination-based Fruit Fly Optimization Alogithm to find the optimum for a given
        optimization problem.
        Parameters
        ----------
        problem: optimization object
            Object containing fitness function optimization problem to be solved.
            For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
            :code:`TSPOpt()`.
        pop_size: int, default: 200
            Size of population to be used in genetic algorithm.
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
            plt.axis()

        # Initialize problem, population and attempts counter
        problem.reset()
        problem.random_pop(pop_size)
        attempts = 0
        iters = 0

        currentBest = problem.best_child()

        while (attempts < max_attempts) and (iters < max_iters):
            iters += 1

            # Create next generation of population
            next_gen = []

            for i in range(pop_size):
                # Select parents
                fly = problem.get_population()[i]
                distances = problem.fitness_fn.distances
                coords = problem.fitness_fn.coords
                if coords != None:
                    newfly = EFOA.reverseOperatorCoords(fly, coords)
                else:
                    newfly = EFOA.reverseOperatorDists(fly, distances)
                if (problem.eval_fitness(newfly) > problem.eval_fitness(fly)):
                    next_gen.append(newfly)
                else:
                    next_gen.append(fly)
            next_gen = np.array(next_gen)
            problem.set_population(next_gen)
            next_gen = []
            best = problem.best_child()
            if problem.eval_fitness(best) > problem.eval_fitness(currentBest):
                currentBest = best
            for i in range(pop_size):
                # Select parents
                fly = problem.get_population()[i]
                if (fly != currentBest).all():
                    newfly = EFOA.multiplicationOperator(fly, currentBest, problem)
                    next_gen.append(newfly)
                else:
                    next_gen.append(fly)
            next_gen = np.array(next_gen)
            sorted_next = sorted(next_gen, key=problem.eval_fitness,reverse=True)
            last = round((1-p) * pop_size) + 1
            next_gen = sorted_next[:last]
            toAdd = [problem.random() for _ in range(pop_size - len(next_gen))]
            if len(toAdd) != 0:
                next_gen = np.concatenate((next_gen, toAdd))
            problem.set_population(next_gen)
            next_gen = []

            next_state = problem.best_child()
            next_fitness = problem.eval_fitness(next_state)

            if next_fitness > problem.eval_fitness(currentBest):
                currentBest = next_state

            # If best child is an improvement,
            # move to that state and reset attempts counter
            if next_fitness > problem.get_fitness():
                problem.set_state(next_state)
                attempts = 0

            else:
                attempts += 1

            if curve:
                fitness_curve.append(problem.get_fitness())
                plt.plot(fitness_curve)
                plt.pause(0.0001)

        best_fitness = problem.get_maximize()*problem.get_fitness()
        best_state = problem.get_state()

        if curve:
            return best_state, best_fitness, np.asarray(fitness_curve)

        return best_state, best_fitness  


class ExactSolver:
    def bruteforce(problem):
        """Use exact solver to find the optimum for a given
        optimization problem.
        Parameters
        ----------
        problem: optimization object
            Object containing fitness function optimization problem to be solved.
            For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
            :code:`TSPOpt()`.
        Returns
        -------
        best_state: array
            Numpy array containing state that optimizes the fitness function.
        best_fitness: float
            Value of fitness function at best state.
        """
        best_state = list(np.arange(problem.length))
        best_fitness = -np.inf
        states = list(permutations(best_state))
        for state in states:
            fitness = problem.eval_fitness(state)
            if fitness > best_fitness:
                best_fitness = fitness
                best_state = state
        return best_state, problem.get_maximize() * best_fitness

    def mip(problem):
        from mip import Model, xsum, minimize, BINARY

        # number of nodes and list of vertices
        n, V = problem.length, set(range(problem.length))

        model = Model()

        # binary variables indicating if arc (i,j) is used on the route or not
        x = [[model.add_var(var_type=BINARY) for _ in V] for _ in V]

        # continuous variable to prevent subtours: each city will have a
        # different sequential id in the planned route except the first one
        y = [model.add_var() for _ in V]

        # objective function: minimize the distance
        model.objective = minimize(xsum(problem.fitness_fn.get_path_length(i,j)*x[i][j] for i in V for j in V))

        # constraint : leave each city only once
        for i in V:
            model += xsum(x[i][j] for j in V - {i}) == 1

        # constraint : enter each city only once
        for i in V:
            model += xsum(x[j][i] for j in V - {i}) == 1

        # subtour elimination
        for (i, j) in product(V - {0}, V - {0}):
            if i != j:
                model += y[i] - (n+1)*x[i][j] >= y[j]-n

        # optimizing
        model.optimize()

        best_state = np.zeros(problem.length, dtype=np.int32)

        # checking if a solution was found
        if model.num_solutions:
            print('route with total distance %g found: %s'
                    % (model.objective_value, 0))
            nc = 0
            index = 1
            while True:
                nc = [i for i in V if x[nc][i].x >= 0.99][0]
                print(' -> %s' % nc)
                if nc == 0:
                    break
                best_state[index]=nc
                index += 1
        
        best_fitness = problem.eval_fitness(best_state)
        return best_state, best_fitness
