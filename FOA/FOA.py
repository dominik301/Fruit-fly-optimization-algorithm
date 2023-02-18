import numpy as np
import matplotlib.pyplot as plt

def Fitness(x):
    return x**2 - 5

def FOA(maxgen,sizepop):
    # Random initial fruit fly location
    X_axis = 10 * np.random.rand()
    Y_axis = 10 * np.random.rand()

    # Fruit flies begin their search for excellence, using their sense of smell to find food
    X = []
    Y = []
    D = []
    S = []
    Smell = []
    for i in range(sizepop):
        # Giving Drosophila individuals the ability to search for food using olfaction in random directions and distances
        X.append(X_axis + 2 * np.random.rand() - 1)
        Y.append(Y_axis + 2 * np.random.rand() - 1)

        # Since the location of the food is not known, the distance to the origin (Dist) is estimated first, and then the taste concentration determination value (S) is calculated, which is the inverse of the distance
        D.append((X[i]**2 + Y[i]**2)**0.5)
        S.append(1 / D[i])

        # The taste concentration determination value (S) is substituted into the taste concentration determination function (or Fitness function) to find the taste concentration (Smell(i)) at the location of the individual fruit fly.
        Smell.append(Fitness(S[i]))

    # Identify the Drosophila with the lowest flavor concentration in this Drosophila population (find the minimum value)
    bestSmell, bestindex = min(Smell),Smell.index(min(Smell))

    # Retain the best flavor concentration value with the coordinates of x, y. At this time, the fruit flies in the swarm use vision to fly to the location
    X_axis = X[bestindex]
    Y_axis = Y[bestindex]
    Smellbest = bestSmell

    # Drosophila iterative merit search begins
    yy = []
    Xbest = []
    Ybest = []
    for g in range(maxgen):
        # Giving Drosophila individuals the ability to search for food using olfaction in random directions and distances
        for i in range(sizepop):
            # Giving Drosophila individuals the ability to search for food using olfaction in random directions and distances
            X[i] = X_axis + 2 * np.random.rand() - 1
            Y[i] = Y_axis + 2 * np.random.rand() - 1

            # Since the location of the food is not known, the distance to the origin (Dist) is estimated first, and then the taste concentration determination value (S) is calculated, which is the inverse of the distance
            D[i] = (X[i]**2 + Y[i]**2)**0.5
            S[i] = 1 / D[i]

            # The taste concentration determination value (S) is substituted into the taste concentration determination function (or Fitness function) to find the taste concentration (Smell(i)) at the location of the individual fruit fly.
            Smell[i] = Fitness(S[i])

        # Identify the Drosophila with the lowest flavor concentration in this Drosophila population (find the minimum value)
        bestSmell, bestindex = min(Smell),Smell.index(min(Smell))

        # Determine whether the flavor concentration is better than the previous iteration of flavor concentration, and if so, keep the best flavor concentration value with the coordinates of x, y. At this time, the fruit fly population uses vision to fly to that location
        if bestSmell < Smellbest:
            X_axis = X[bestindex]
            Y_axis = Y[bestindex]
            Smellbest = bestSmell

        # Each optimal Smell value is recorded into the yy array and the optimal iteration coordinates are recorded
        yy.append(Smellbest)
        Xbest.append(X_axis)
        Ybest.append(Y_axis)

    return yy, Xbest, Ybest

maxgen = 200
sizepop = 50
yy, Xbest, Ybest = FOA(maxgen,sizepop)
ax1 = plt.subplot(121)
ax1.plot(yy)
ax1.set(xlabel = 'Iteration Number',ylabel = 'Smell',title = 'Optimization process')
ax2 = plt.subplot(122)
ax2.plot(Xbest,Ybest)
ax2.set(xlabel = 'X-axis',ylabel = 'Y-axis',title = 'Fruit fly flying route')
plt.show()

