import numpy as np
import matplotlib.pyplot as plt
import time
from foa.problemTypes import *
import datetime
import pandas as pd
from hyperopt import hp, fmin, rand, Trials, space_eval, STATUS_OK, tpe
from hyperopt.mongoexp import MongoTrials
import multiprocessing
import sys

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
        return best_state, np.mean(fitnesses)
    
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
            #self.problemType.plot(best_state, FUNCTIONS[fn.__name__] + ", Länge: " + str(best_fitness))

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

    def tune(self, n=10, vec_pop_size=[50,100,200], vec_V_r=[0.1,0.3,0.5], vec_NN=[5,10,15]):
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

    def tune_efoa(self, n=10, vec_pop_size=[50,100,200], vec_attempts=[25,50,100], vec_p=[0.05,0.1,0.2]):
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

    def tune_ga(self, n=10, vec_pop_size=[50,100,200], vec_attempts=[25,50,100], vec_mutation_prob=[0.1,0.2,0.4]):
        for pop_size in vec_pop_size:
            print("Pop size: ", pop_size)
            _, best_fitness = self.simulate(self.problemType.solve_ga, n, name=str(pop_size), pop_size=pop_size)

            print("Best fitness: ", best_fitness)

        self.plotStatistics([str(pop_size) for pop_size in vec_pop_size], testCase="GA Popsize")
        self.results = {}
        
        for attempts in vec_attempts:
            print("Attempts: ", attempts)
            _, best_fitness = self.simulate(self.problemType.solve_ga, n, name=str(attempts), max_attempts=attempts)

            print("Best fitness: ", best_fitness)
        
        self.plotStatistics([str(attempts) for attempts in vec_attempts], testCase="GA Attempts")
        self.results = {}
        for p in vec_mutation_prob:
            print("p: ", p)
            _, best_fitness = self.simulate(self.problemType.solve_ga, n, name=str(p), mutation_prob=p)

            print("Best fitness: ", best_fitness)

        self.plotStatistics([str(p) for p in vec_mutation_prob], testCase="GA mutation probabiliy")
        self.results = {}

    def tune_sa(self, n=10, vec_attempts=[25,50,100], vec_schedule=[mlrose.GeomDecay, mlrose.ExpDecay, mlrose.ArithDecay]):
        '''schedule: schedule object, default: :code:`mlrose.GeomDecay()`
            Schedule used to determine the value of the temperature parameter.
        max_attempts: int, default: 10
            Maximum number of attempts to find a better neighbor at each step.
        max_iters: int, default: np.inf
            Maximum number of iterations of the algorithm.'''
        
        for attempts in vec_attempts:
            print("Attempts: ", attempts)
            _, best_fitness = self.simulate(self.problemType.solve_sa, n, name=str(attempts), max_attempts=attempts)

            print("Best fitness: ", best_fitness)
        
        self.plotStatistics([str(attempts) for attempts in vec_attempts], testCase="SA Attempts")
        self.results = {}

        for schedule in vec_schedule:
            print("Schedule: ", schedule.__name__)
            _, best_fitness = self.simulate(self.problemType.solve_sa, n, name=schedule.__name__, schedule=schedule())

            print("Best fitness: ", best_fitness)

        self.plotStatistics([schedule.__name__ for schedule in vec_schedule], testCase="SA Schedule")
        self.results = {}

    def objective(self, fn, args):
        # define an objective function
        start = time.time()
        _, best_fitness = self.simulate(fn, 3, **args)
        return {'loss': best_fitness, 'eval_time': time.time() - start, 'status': STATUS_OK, 'other_stuff': {'args': args}}
    
    def objective_foa(self, args):
        return self.objective(self.problemType.solve, args)
    
    def objective_efoa(self, args):
        return self.objective(self.problemType.solve_efoa, args)
    
    def objective_ga(self, args):
        return self.objective(self.problemType.solve_ga, args)
    
    def objective_sa(self, args):
        return self.objective(self.problemType.solve_sa, args)

    def tune_random(self, alg='SA', debug=False):
        # define a search space
        if alg == 'FOA':
            space = {
                'NN': 1 + hp.randint('NN', 19),
                'V_r': hp.uniform('V_r', 0.1, 0.5),
                'pop_size': 50 + hp.randint('pop_size', 450),
                'max_attempts': 1 + hp.randint('max_attempts', 99),
            }
            fn = self.objective_foa
        elif alg == 'EFOA':
            space = {
                'p': hp.uniform('p', 0, 0.2),
                'pop_size': 50 + hp.randint('pop_size', 450),
                'max_attempts': 1 + hp.randint('max_attempts', 99)
            }
            fn = self.objective_efoa
        elif alg == 'GA':
            space = {
                'pop_size': 50 + hp.randint('pop_size', 450),
                'mutation_prob': hp.uniform('mutation_prob', 0, 0.5),
                'max_attempts': 1 + hp.randint('max_attempts', 99)
            }
            fn = self.objective_ga
        elif alg == 'SA':
            space = {
                'schedule': hp.choice('schedule', ["GeomDecay", "ExpDecay", "ArithDecay"]),
                'max_attempts': 1 + hp.randint('max_attempts', 99)
            }
            fn = self.objective_sa
        else:
            raise Exception('Unknown algorithm: ' + alg)

        # minimize the objective over the space
        trials = MongoTrials('mongo://localhost:1234/foo2_db/jobs', exp_key=alg+ str(2))
        best = fmin(fn, space, algo=rand.suggest, max_evals=50, trials=trials)

        #print(space_eval(space, best))
        
        results = []
        for result in trials.results:
            results.append([result['loss'], result['eval_time'], result['other_stuff']['args']])
        results = sorted(results, key=lambda x: [x[0], x[1]])
        
        if debug:
            rows  = []
            for value in results[0][2].keys():
                rows.append(str(value))
            rows.append('loss')
            rows.append('eval_time')
            print(rows)
            for result in results:
                solution = ""
                for value in result[2].values():
                    if isinstance(value, float):
                        solution += "{:.2f}".format(value) + " & "
                    else:
                        solution += str(value) + " & "
                solution += str(int(result[0])) + " & " + "{:.2f}".format(result[1]) + " \\\\"
                print(solution)
        return np.array(results)

def runTest(testCase):
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
            solver.problemType.plot(best_state, title=v.__name__)
        solver.plotStatistics([v.__name__ for v in V], x_axis=['V1', 'V3'], testCase=testCase)
    elif testCase == "HYPERPARAMETER_TUNING":
        tuner.tune()
        tuner.tune_efoa()
        #tuner.tune_sa()
        #tuner.tune_ga()
        '''for alg in ['FOA', 'EFOA', 'GA', 'SA']:
            results = tuner.tune_random(alg)
            plt.scatter(results[:,1],results[:,0], label=alg)
            ax = plt.gca()
        plt.xlabel('Zeit (s)')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()'''
        if False:
            for x in rows[:-2]:
                print([result[x] for result in results[:,2]])
                print(results[:,0])
                plt.scatter([result[x] for result in results[:,2]],results[:,0])
                ax = plt.gca()
                plt.xlabel(x)
                plt.ylabel('Fitness')
                ax.set_title('Fitness vs. ' + x)
                plt.show()
    else:
        functions = [solver.problemType.solve_efoa, solver.problemType.solve, solver.problemType.solve_ga, solver.problemType.solve_sa]
        solver.solve_for_fn(functions, n=10)
        solver.plotStatistics([fn.__name__ for fn in functions], x_axis=['EFOA', 'FOA', 'GA', 'SA'], testCase=testCase)
        
    
if __name__ == "__main__":
    testCase = "RACK"
    
    if len(sys.argv) > 1:
        testCase = sys.argv[1]
    if testCase == "ALL":
        pool_obj = multiprocessing.Pool()
        pool_obj.map(runTest, TEST_CASES)
    elif testCase not in TEST_CASES:
        raise Exception('Unknown test case: ' + testCase)
    else:
        runTest(testCase)
