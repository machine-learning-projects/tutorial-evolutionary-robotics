# core03 - Evolving a Neural Network - https://www.uvm.edu/~ludobots/index.php/ER/Assignment3

########################################################################################################################

# Project Description
# In this project you will apply the hillclimber you developed in this project to the artificial neural networks (ANNs)
# you developed in this project. Below is a summary of what you will implement in this project.

# 1. Create a (numNeurons=)10 x (numNeurons=)10 matrix called parent to hold the synapse weights for an ANN with
# numNeurons=10 neurons.

# 2. Randomize the parent synapse matrix.

# 3. Create a (numUpdates=)10 x (numNeurons=)10 matrix called neuronValues to hold the values of the neurons as the
# network is updated. The first row stores the initial values of the neurons, which for this project will all initially\
# be set to 0.5. The second row will store the new values of each neuron, and so on.

# 4. Create a vector of length 10 called desiredNeuronValues that holds the values that each neuron in the ANN should
# reach. We'll compare the final values of the neurons-the last row of neuronValues-to this vector. The closer the
# match, the higher the fitness of the synapse matrix.

# 5. Update neuronValues nine times (thus filling in rows two through 10) using the parent synapse weights.

# 6. The program will then loop through 1000 generations. For each loop: (1) Create the child synapse matrix by copying
# and perturbing the parent synapse matrix. (2) Set each element in the first row of neuronValues to 0.5. (3) Update the
# neuronValues of the ANN nine times using the child synapse values. (4) Calculate and save the fitness of the child
# synapses as childFitness. (5) If the childFitness is better than parentFitness, replace the parent synapse matrix with
# the child synapse matrix and set parentFitness to the childFitness value.

########################################################################################################################

# For first fitness, want neurons to alternate between on/off
# For second fitness, want neurons to alternate between on/off at each time-step, and to be as different as possible
# from their neighbors, giving a checkerboard pattern

import random
from copy import deepcopy

import matplotlib.cm as cm
import matplotlib.pyplot as neuronValuePlot
import matplotlib.pyplot as fitnessPlot
import numpy as np

numUpdates = 10
numNeurons = 10


# Project Details
# Here are the steps to implement the program:
# 1. Back up your Python code from the previous project. Encapsulate your code from the previous project in a single
# file, e.g. Project_ANN.py, such that when you run it you can reproduce all of the images from that project. This will
# prove to you that your code is working fine, as you will use it in this and subsequent projects.

# 2. Create a blank Python file called Project_Evolve_ANNs.py. As you implement this project, copy and paste functions
# from the two previous projects as they are needed

# Return a rows by columns matrix with all elements set to zero
def MatrixCreate(rows, columns):
    return np.zeros(shape=(rows, columns))


# 5. Modify the MatrixRandomize function so that it returns values in [-1,1] rather than [0,1]. parent now encodes
# synaptic weights for a neural network with 10 neurons. print parent to make sure this was done correctly. Also,
# update MatrixPerturb function, so it is able to return new values in the range [-1, 1], rather than [0, 1].

# Put random values drawn from [-1, 1] into each element of vector v
def MatrixRandomize(v):
    for x in np.nditer(v, op_flags=['readwrite']):
        x[...] = random.uniform(-1, 1)
    return v


# Makes a copy of the parent vector p, and then considers each element in the new vector c
# With probability prob, the element's value is replaced with a new random value drawn from [0, 1]
# otherwise, the element's value is left as is
def MatrixPerturb(p, prob):
    c = deepcopy(p)
    for x in np.nditer(c, op_flags=['readwrite']):
        if prob > random.random():
            x[...] = random.uniform(-1, 1)

    return c


def Update(neuronValues, synapses, i):
    for j in xrange(len(neuronValues[i])):
        temp = 0
        for k in xrange(0, 10):
            a_k = neuronValues[i - 1][k]
            w_jk = synapses[k][j]
            temp += a_k * w_jk

        if temp < 0:
            temp = 0
        elif temp > 1:
            temp = 1
        neuronValues[i][j] = temp

    return neuronValues


# 13. VectorCreate returns a row vector of length 10. To create this function, copy, rename and modify MatrixCreate. To
# create a vector using NumPy: v = zeros((width), dtype='f'). The for loop counts from 1 to 10 in steps of 2: 1,3,5,...
# This code should produce a vector of the form [0,1,0,1,0,1,0,1,0,1].
def VectorCreate(width):
    return np.zeros((width), dtype='f')


# 14. Now, create a function MeanDistance(v1,v2) that returns the normalized distance between two vectors with elements
# between 0 and 1: The function should return 0 if the vectors are the same, and 1 if they are maximally different. I
# would suggest you use mean squared error , which here would be
# d = ( (v1[0]-v2[0])2 + ... + (v1[9]-v2[9])2 ) / 10
# This function should be used to compute the distance d between actualNeuronValues and desiredNeuronValues. Fitness
# should then return f = 1 - d: the lower the distance between the vectors, the closer the fitness approaches 1.
def MeanDistance(v1, v2):
    if (v1 == v2).all():  # check if vectors are the same
        return 0

    d = 0
    for i in xrange(len(v1)):
        d += (v1[i] - v2[i]) ** 2
    d /= 10

    return d


# 7. You must now modify the Fitness function so that it returns a single value that indicates the fitness, or quality
# of the neural network specified by the synaptic weights stored in parent. Steps 7 through 14 will help you to do this.
# First, delete all the code currently within the Fitness function, and add code into the now-empty function that
# creates an all-zero (numUpdates=)10 x (numNeurons=)10 matrix neuronValues like you did in the ANN project. Print the
# matrix to make sure it is correct.

# Want to converge with even/odd neurons alternating on/off

# Returns the mean value of all of the elements in vector v
def Fitness(v):
    neuronValues = MatrixCreate(numUpdates, numNeurons)

    # 8. Fill each element in the first row of neuronValues with 0.5: each of the 10 neurons will start with this value.
    # print neuronValues to make sure this was done correctly.
    for x in np.nditer(neuronValues[0], op_flags=['readwrite']):
        x[...] = 0.5

    # 9. Copy across the Update function you developed in the ANN project, step 12. Use it to fill in the second row of
    # neuronValues using the synaptic weights stored in parent. Apply it again to fill in the third row. Apply it nine
    # times in total so that neuronValues is completely filled (numUpdates=10 in this project). Print neuronValues
    # periodically to make sure it is being updated correctly (i.e. each row of zeros are replaced by non-zero values).
    for i in xrange(1, numUpdates):
        Update(neuronValues, v, i)

    # 11. Now we are going to calculate how close the final set of neuron values is to some desired set of neuron
    # values.After neuronValues has been filled, extract the last row as follows:
    # actualNeuronValues = neuronValues[9,:]. This copies the last row of neuronValues.
    actualNeuronValues = neuronValues[9, :]

    # 12. Create a vector that stores the desired values for the neurons. Let's select for odd-numbered neurons that are
    # on, and even-numbered neurons that are off:
    desiredNeuronValues = VectorCreate(10)
    for j in range(1, 10, 2):
        desiredNeuronValues[j] = 1

    distance = MeanDistance(actualNeuronValues, desiredNeuronValues)

    return 1 - distance


# 20. Now copy the Fitness function, rename it Fitness2, and change the two calls to Fitness in your main function to
# calls to Fitness2.

# More difficult: want to converge on neurons being as different from their neighbors as possible, and to alternate each
# time-step

# Returns the mean value of all of the elements in vector v
def Fitness2(v):
    neuronValues = MatrixCreate(numUpdates, numNeurons)

    for x in np.nditer(neuronValues[0], op_flags=['readwrite']):
        x[...] = 0.5

    for i in xrange(1, numUpdates):
        Update(neuronValues, v, i)

    # 21. Leave the internals of Fitness2 as they are, but change how the fitness of the neural network's behavior is
    # calculated. Remove the mean squared error calculation, and instead compute the average difference between
    # neighboring elements in the matrix:
    # diff=0.0
    # for i in range(1,9):
    #       for j in range(0,9):
    #            diff=diff + abs(neuronValues[i,j]-neuronValues[i,j+1])
    #            diff=diff + abs(neuronValues[i+1,j]-neuronValues[i,j])
    # diff=diff/(2*8*9)
    # Note that in calculating our fitness, we should ignore the top row of our matrix, which was initialized with all
    # values set to 0.5.
    diff = 0.0
    for i in range(1, 9):
        for j in range(0, 9):
            diff = diff + abs(neuronValues[i, j] - neuronValues[i, j + 1])
            diff = diff + abs(neuronValues[i + 1, j] - neuronValues[i, j])

    diff = diff / (2 * 8 * 9)

    return diff


def plotGenes(matrix):
    neuronValuePlot.clf()
    neuronValuePlot.imshow(matrix, cmap=cm.gray, aspect='auto', interpolation='nearest')
    neuronValuePlot.show()


def plotFitness(fits):
    fitnessPlot.plot(fits)
    fitnessPlot.show()


# 3. Copy and paste the main function that you created in the Hillclimber project at step 7, comment out all but the
# first two lines and change it so that instead of creating a vector of random numbers, you create a matrix of random
# numbers. This matrix contains the synaptic weights of the parent neural network. (You'll have to copy across the
# MatrixCreate and MatrixRandomize functions from that project as well.) Note that in Python, putting a hash symbol (#)
# at the front of the line comments it out:
parent = MatrixCreate(10, 10)
parent = MatrixRandomize(parent)
# 6. Uncomment the third line, and copy across the Fitness function you created in the Hillclimber project.
parentFitness = Fitness2(parent)

# 17. Just before the main loop begins, create a neuronValues matrix and fill its top row with 0.5 activations. Then use
# the parent network to fill all the lower rows of neuronValues by means of nine Updates. Send the neuronValues as the
# argument to your matrix imaging function from the previous project. {see core02 step 13}. Copy and paste the resulting
# image into your document, which should look like Fig. 1a or d. Note that the image should show a band of gray at the
# top, which corresponds to the initial 0.5 settings of all the neurons.
neuronValues = MatrixCreate(numUpdates, numNeurons)
for x in np.nditer(neuronValues[0], op_flags=['readwrite']):
    x[...] = 0.5

for i in xrange(1, numUpdates):
    Update(neuronValues, parent, i)

plotGenes(neuronValues)

# 15. Now, uncomment the remaining lines of the main function, and ensure that your hillclimber is working correctly:
# child should be a slightly different matrix compared to parent, and parentFitness should increase toward 1 as the loop
# runs
fits = VectorCreate(1000)
for currentGeneration in range(0, 1000):
    print currentGeneration, parentFitness
    child = MatrixPerturb(parent, 0.05)
    childFitness = Fitness2(child)

    if (childFitness > parentFitness):
        parent = child
        parentFitness = childFitness

    # 16. Add a fitness vector to the main function, and record parentFitness into this vector after each pass through
    # the loop
    fits[currentGeneration] = parentFitness

# 4. Insert a print parent statement after the two lines to make sure the matrix was randomized correctly.
# print parent

# 18. After the main loop has finished, express the behavior of the parent by nine Updates of an initialized
# neuronValues. It will begin with 0.5 activations at the top row. Add another call to the matrix imaging function
# (sending the resulting neuronValues as argument) to show the behavior of the final, evolved neural network. Save this
# image. It should show an alternating pattern of black and white pixels in the bottom row, like Fig. 1b. It may not be
# perfectly alternating if the fitness of that run did not get to a fitness of 1; this is fine if this is the case.
neuronValues = MatrixCreate(numUpdates, numNeurons)
for x in np.nditer(neuronValues[0], op_flags=['readwrite']):
    x[...] = 0.5

for i in xrange(1, numUpdates):
    Update(neuronValues, parent, i)

plotGenes(neuronValues)

# 19. Store the fitness of the parents as the main loop iterates in a vector, and plot that vector. It should look like
# Fig. 1c. Save your resulting image. Run the program a few times to see what kinds of patterns you get.
plotFitness(fits)

# 22. Re-run the hillclimber, and save out the initial random neural network's behavior (as in Fig. 1d), the behavior of
# the final, evolved neural network (as in Fig. 1e), and the fitness increase during the hillclimber (as in Fig. 1f).
# Note that the evolved neural network may not reach a perfect checkerboard configuration; that is fine if this is the
# case. Run the program a few times to see what kinds of patterns you get.

# Things to think about: What other kinds of neural behavior could you select for? What visual pattern would it produce
# when plotted? If you like, try implementing these new fitness functions and see whether you get the pattern you were
# expecting.
