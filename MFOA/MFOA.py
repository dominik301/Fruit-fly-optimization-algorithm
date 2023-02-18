from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

def FF(X):
    # return (4-2.1*(X[0]**2)+(X[0]**4)/3)*(X[0]**2)+X[0]*X[1]+(-4+4*(X[1]**2))*(X[1]**2)
    return (X[0])**2 + X[1]**2
def MFOA(kmax,M,Popsize,dim,a,b,phi):
    """
    Multigroup Drosophila search algorithm
    kmax: maximum number of iterations
    M: number of subpopulations
    Popsize: subpopulation population
    dim: Dimensions of the search space
    a,b: the upper and lower interval of each search dimension, here first set to be the same for each dimension
    phi: iteration parameter in the algorithm
    """

    X = np.empty((M,Popsize,dim),dtype='float32')# the position of each fruit fly in space
    Smell = np.empty((M,Popsize),dtype='float32')# the smell of each fruit fly in the space
    bestSmellm = np.empty(M,dtype='float32')# the best smell for each population for a given iteration
    bestIndexm = np.empty(M,dtype='int')# the index of the best smell of each population in the population for a given iteration
    Smellbestm = np.empty(M,dtype='float32')# the best smell of all populations in the current iteration
    X_axism = np.empty((M,dim),dtype='float32')# current iteration of the position of the best smell in all populations
    Smellbest = np.empty(1,dtype='float32')
    X_axis = np.empty(dim,dtype='float32')
    # Loop Iteration
    for k in range(kmax):
        R = ((b-a)/2)*(((kmax-k)/kmax)**phi)
        # Interpopulation cycles
        for m in range(M):
            # Inter-individual cycles
            for i in range(Popsize):
                X[m,i] = X_axism[m] + R * np.random.rand(dim)
                Smell[m,i] = FF(X[m,i])
            bestSmellm[m], bestIndexm[m] = np.min(Smell[m]),np.argmin(Smell[m])# np.argmin()Returns only the index of the first occurrence of the maximum value
            Smellbestm[m] = bestSmellm[m]
            X_axism[m] = X[m,bestIndexm[m]]
            if Smellbestm[m] < Smellbest:
                Smellbest = Smellbestm[m]
                X_axis = X_axism[m]
            X_new = np.mean(X_axism,axis=0)
            if FF(X_new) < Smellbest:
                Smellbest = FF(X_new)
                X_axis = X_new

    return Smellbest,X_axis,X_axism

Smellbest, X_axis,X_axism = MFOA(1000,2,30,2,20,-20,6)
print(Smellbest)
print(X_axis)
print(X_axism)

# x = np.linspace(-2,2,100)
# y = np.linspace(-2,2,100)
# X,Y = np.meshgrid(x,y)
# Z = F1(X,Y)
#
# fig = plt.figure()
# ax =plt.axes(projection='3d')
# ax.plot_wireframe(X,Y,Z)
# plt.show()
# maxgen = 200
# sizepop = 50
# yy, Xbest, Ybest = FOA(maxgen,sizepop)
# ax1 = plt.subplot(121)
# ax1.plot(yy)
# ax1.set(xlabel = 'Iteration Number',ylabel = 'Smell',title = 'Optimization process')
# ax2 = plt.subplot(122)
# ax2.plot(Xbest,Ybest)
# ax2.set(xlabel = 'X-axis',ylabel = 'Y-axis',title = 'Fruit fly flying route')
# plt.show()

