import numpy as np
import matplotlib.pyplot as plt
import time
from problemTypes import *

TEST_CASES = ["RACK","WAREHOUSE","WAREHOUSE_WITH_AISLES","WAREHOUSE_ONE_DIRECTION","HYPERPARAMETER_TUNING", "VISION"]

class Solver:
    def __init__(self, problemType=None, nRows=10, lotsPerRow=10, aisles=[5], count=10):
        if problemType == None:
            self.problemType = Warehouse(nRows, lotsPerRow)
        elif problemType == WarehouseWithAisles:
            self.problemType = WarehouseWithAisles(nRows, lotsPerRow, aisles)
        else:
            self.problemType = problemType(nRows, lotsPerRow)
        self.results = {}
        self.count = count
    
    def simulate(self, fn, n, debug=False, name=None, **kwargs):
        """Simulate a function call n times and return the average time."""
        times = []
        fitnesses = []
        best_state = []
        best_fitness = np.inf
        
        if name == None:
            name = fn.__name__

        for _ in range(n):
            start = time.time()
            state, fitness = fn(count=self.count,**kwargs)
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
        self.results[name] = [np.mean(fitnesses), np.std(fitnesses), np.mean(times), np.std(times)]
        return best_state, best_fitness
    
    def heuristics(self):
        for fn in [self.problemType.midpoint, self.problemType.sshape]:
            print(fn.__name__)
            best_state, best_fitness = self.simulate(fn, 1)

            print("Best fitness: ", best_fitness)
            self.problemType.plot(best_state)

    def solve_for_fn(self, fn, n=5):
        for fn in functions:
            print(fn.__name__)
            best_state, best_fitness = self.simulate(fn, n)
            print("Best fitness: ", best_fitness)
            self.problemType.plot(best_state)

    def plotStatistics(self, rows, x_axis=None):
        if x_axis == None:
            x = rows
        else:
            x = x_axis
        y = [self.results[entry][0] for entry in rows]
        err = [self.results[entry][1] for entry in rows]
        times = [self.results[entry][2] for entry in rows]
        timeerr = [self.results[entry][3] for entry in rows]
        plt.subplot(121)
        
        # Plot scatter here
        plt.bar(x, y)
        plt.errorbar(x, y, yerr=err, fmt="o", color="r")

        plt.subplot(122)

        plt.bar(x, times)
        plt.errorbar(x, times, yerr=timeerr, fmt="o", color="r")
        plt.show()

class HyperParameterTuning(Solver):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def tune(self, n=5, vec_pop_size=[100,300], vec_V_r=[0.1,0.5], vec_NN=[5,15]):
        for pop_size in vec_pop_size:
            print("Pop size: ", pop_size)
            _, best_fitness = self.simulate(self.problemType.solve, n, name=str(pop_size), pop_size=pop_size)

            print("Best fitness: ", best_fitness)

        self.plotStatistics([str(pop_size) for pop_size in vec_pop_size])
        
        for V_r in vec_V_r:
            print("V_r: ", V_r)
            _, best_fitness = self.simulate(self.problemType.solve, n, name=str(V_r), V_r=V_r)

            print("Best fitness: ", best_fitness)
        
        self.plotStatistics([str(V_r) for V_r in vec_V_r])
        
        for NN in vec_NN:
            print("NN: ", NN)
            _, best_fitness = self.simulate(self.problemType.solve, n, name=str(NN), NN=NN)

            print("Best fitness: ", best_fitness)
        
        self.plotStatistics([str(NN) for NN in vec_NN])

    def tune_efoa(self, n=5, vec_pop_size=[100,300]):
        for pop_size in vec_pop_size:
            print("Pop size: ", pop_size)
            _, best_fitness = self.simulate(self.problemType.solve_efoa, n, name=str(pop_size), pop_size=pop_size)

            print("Best fitness: ", best_fitness)

        self.plotStatistics([str(pop_size) for pop_size in vec_pop_size])

if __name__ == "__main__":
    testCase = "VISION"

    if testCase == "RACK":
        solver = Solver(Rack, nRows=10, lotsPerRow=30, count=10)
    elif testCase == "WAREHOUSE":
        solver = Solver(Warehouse, nRows=20, lotsPerRow=50, count=20)
    elif testCase == "WAREHOUSE_WITH_AISLES":
        solver = Solver(WarehouseWithAisles, nRows=10, lotsPerRow=30, aisles=[10,20], count=20)
    elif testCase == "WAREHOUSE_ONE_DIRECTION":
        solver = Solver(WarehouseOneDirection, nRows=20, lotsPerRow=50, count=20)
    elif testCase == "HYPERPARAMETER_TUNING":
        tuner = HyperParameterTuning(nRows=20, lotsPerRow=50, count=20)
    elif testCase == "VISION":
        solver = Solver(Warehouse, nRows=20, lotsPerRow=50, count=20)

    if not testCase == "HYPERPARAMETER_TUNING" and not testCase == "VISION":
        functions = [solver.problemType.solve, solver.problemType.solve_efoa, solver.problemType.solve_ga, solver.problemType.solve_sa] #, solver.problemType.midpoint, solver.problemType.sshape]
        solver.solve_for_fn(functions, n=5)
        solver.plotStatistics([fn.__name__ for fn in functions], x_axis=['FOA', 'EFOA', 'GA', 'SA'])#, 'Mittelpunkt', 'S-Form'])
    elif testCase == "VISION":
        V = [FOA.v1, FOA.v3]
        for v in V:
            best_state, best_fitness = solver.simulate(solver.problemType.solve, n=5,name=v.__name__, visionFn=v)
            print("Best fitness: ", best_fitness)
            solver.problemType.plot(best_state)
        solver.plotStatistics([v.__name__ for v in V], x_axis=['V1', 'V3'])
    else:
        tuner.tune_efoa()
        
    