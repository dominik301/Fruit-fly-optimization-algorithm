import numpy as np
import matplotlib.pyplot as plt
import time
from problemTypes import *
import datetime
import pandas as pd

TEST_CASES = ["RACK","WAREHOUSE","WAREHOUSE_WITH_AISLES","WAREHOUSE_ONE_DIRECTION","HYPERPARAMETER_TUNING", "VISION"]
FUNCTIONS = {"solve_efoa": "EFOA", "solve": "FOA", "midpoint": "Mittelpunkt", "sshape": "S-Form", "solve_ga": "GA", "solve_sa": "SA"}
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

        if debug:
            print("Fitness: ", np.mean(fitnesses), "+/-", np.std(fitnesses))
            print("Time: ", np.mean(times), "+/-", np.std(times))
        self.results[name] = {"fitness": fitnesses, "time": times}
        return best_state, best_fitness
    
    def heuristics(self):
        for fn in [self.problemType.midpoint, self.problemType.sshape]:
            print(fn.__name__)
            best_state, best_fitness = self.simulate(fn, 1)

            print("Best fitness: ", best_fitness)
            self.problemType.plot(best_state)

    def solve_for_fn(self, functions, n=5):
        for fn in functions:
            print(fn.__name__)
            best_state, best_fitness = self.simulate(fn, n)
            print("Best fitness: ", best_fitness)
            self.problemType.plot(best_state, FUNCTIONS[fn.__name__] + ", Länge: " + str(best_fitness))

    def plotStatistics(self, rows=None, x_axis=None, testCase="", show=True):
        if x_axis == None:
            x = rows
        else:
            x = x_axis
        plt.figure(figsize=(14,7))
        ax1 = plt.subplot(121)
        
        fitness = {}
        times = {}
        for key,value in self.results.items():
            times[key] = value["time"]
            fitness[key] = value["fitness"]
        df = pd.DataFrame(fitness)
        plt.boxplot(df)
        plt.xticks(range(1, len(df.columns) + 1), x)
        
        ax1.title.set_text('Fitness')

        ax2 = plt.subplot(122)
        
        df = pd.DataFrame(times)
        plt.boxplot(df)
        plt.xticks(range(1, len(df.columns) + 1), x)
        
        ax2.title.set_text('Zeit')
        if show:
            plt.show()
        else:
            name = testCase + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + ".png"
            plt.savefig(name)
            plt.close()

class HyperParameterTuning(Solver):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def tune(self, n=10, vec_pop_size=[100,300], vec_V_r=[0.1,0.5], vec_NN=[5,15]):
        for pop_size in vec_pop_size:
            print("Pop size: ", pop_size)
            _, best_fitness = self.simulate(self.problemType.solve, n, name=str(pop_size), pop_size=pop_size)

            print("Best fitness: ", best_fitness)

        self.plotStatistics([str(pop_size) for pop_size in vec_pop_size], testCase="FOA Popsize")
        self.results = {}
        
        for V_r in vec_V_r:
            print("V_r: ", V_r)
            _, best_fitness = self.simulate(self.problemType.solve, n, name=str(V_r), V_r=V_r)

            print("Best fitness: ", best_fitness)
        
        self.plotStatistics([str(V_r) for V_r in vec_V_r], testCase="V_r")
        self.results = {}
        
        for NN in vec_NN:
            print("NN: ", NN)
            _, best_fitness = self.simulate(self.problemType.solve, n, name=str(NN), NN=NN)

            print("Best fitness: ", best_fitness)
        
        self.plotStatistics([str(NN) for NN in vec_NN], testCase="NN")
        self.results = {}

    def tune_efoa(self, n=10, vec_pop_size=[100,200,1000], vec_attempts=[10,100], vec_p=[0.05,0.1,0.2]):
        for pop_size in vec_pop_size:
            print("Pop size: ", pop_size)
            _, best_fitness = self.simulate(self.problemType.solve_efoa, n, name=str(pop_size), pop_size=pop_size)

            print("Best fitness: ", best_fitness)

        self.plotStatistics([str(pop_size) for pop_size in vec_pop_size], testCase="EFOA Popsize")
        self.results = {}

        for attempts in vec_attempts:
            print("Attempts: ", attempts)
            _, best_fitness = self.simulate(self.problemType.solve_efoa, n, name=str(attempts), max_attempts=attempts)

            print("Best fitness: ", best_fitness)
        
        self.plotStatistics([str(attempts) for attempts in vec_attempts], testCase="EFOA Attempts")
        self.results = {}

        for p in vec_p:
            print("p: ", p)
            _, best_fitness = self.simulate(self.problemType.solve_efoa, n, name=str(p), p=p)

            print("Best fitness: ", best_fitness)

        self.plotStatistics([str(p) for p in vec_p], testCase="EFOA p")
        self.results = {}

if __name__ == "__main__":
    testCase = "HYPERPARAMETER_TUNING"

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

    if testCase == "WAREHOUSE":
        functions = [solver.problemType.solve_efoa, solver.problemType.solve, solver.problemType.solve_ga, solver.problemType.solve_sa, solver.problemType.midpoint, solver.problemType.sshape]
        solver.solve_for_fn(functions, n=10)
        solver.plotStatistics([fn.__name__ for fn in functions], x_axis=['EFOA', 'FOA', 'GA', 'SA', 'Mittelpunkt', 'S-Form'], testCase=testCase)
    elif testCase == "VISION":
        V = [FOA.v1, FOA.v3]
        for v in V:
            best_state, best_fitness = solver.simulate(solver.problemType.solve, n=10,name=v.__name__, visionFn=v)
            print("Best fitness: ", best_fitness)
            solver.problemType.plot(best_state)
        solver.plotStatistics([v.__name__ for v in V], x_axis=['V1', 'V3'], testCase=testCase)
    elif testCase == "HYPERPARAMETER_TUNING":
        tuner.tune()
        tuner.tune_efoa()
    else:
        functions = [solver.problemType.solve_efoa, solver.problemType.solve, solver.problemType.solve_ga, solver.problemType.solve_sa]
        solver.solve_for_fn(functions, n=10)
        solver.plotStatistics([fn.__name__ for fn in functions], x_axis=['EFOA', 'FOA', 'GA', 'SA'], testCase=testCase)
        
    