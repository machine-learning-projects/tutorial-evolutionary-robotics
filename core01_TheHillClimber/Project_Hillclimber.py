# The Hill Climber - https://www.uvm.edu/~ludobots/index.php/ER/Assignment1
# The hill climber will search for vectors in which all of the elements have values near one.

import random
from copy import deepcopy

import matplotlib.pyplot as fitnessPlot
import numpy as np


# Return a rows by columns matrix with all elements set to zero
def MatrixCreate(rows, columns):
    return np.zeros(shape=(rows, columns))


# Put random values drawn from [0, 1] into each element of vector v
def MatrixRandomize(v):
    for x in np.nditer(v, op_flags=['readwrite']):
        x[...] = random.uniform(0, 1)
    return v


# Returns the mean value of all of the elements in vector v
def Fitness(v):
    return np.mean(v)


# Makes a copy of the parent vector p, and then considers each element in the new vector c
# With probability prob, the element's value is replaced with a new random value drawn from [0, 1]
# otherwise, the element's value is left as is
def MatrixPerturb(p, prob):
    c = deepcopy(p)
    for x in np.nditer(c, op_flags=['readwrite']):
        if prob > random.random():
            x[...] = random.random()

    return c


# Serial Hill Climber
def SerialHillClimber():
    fits = MatrixCreate(1, 5000)

    parent = MatrixCreate(1, 50)
    parent = MatrixRandomize(parent)
    parentFitness = Fitness(parent)
    Genes = MatrixCreate(50, 5000)

    for currentGeneration in range(5000):
        fits[0][currentGeneration] = parentFitness
        child = MatrixPerturb(parent, 0.05)
        childFitness = Fitness(child)
        if childFitness > parentFitness:
            parent = child
            parentFitness = childFitness

        for idx, val in enumerate(parent[0]):
            Genes[idx][currentGeneration] = val

    return fits


def PlotVectorAsLine():
    all_fits = MatrixCreate(5, 5000)
    for i in xrange(5):
        all_fits[i] = SerialHillClimber()

    for i in xrange(4):
        fitnessPlot.plot(all_fits[i])

    fitnessPlot.xlabel('Generation')
    fitnessPlot.ylabel('Fitness')
    fitnessPlot.show()
