# core02 - Your First Neural Network - https://www.uvm.edu/~ludobots/index.php/ER/Assignment2

########################################################################################################################

# Project Description:
# In this project you will be creating an artificial neural network (ANN). There are many kinds of ANNs, but they all
# share one thing in common: they are represented as a directed graph in which the nodes are models of biological
# neurons, and edges are models of biological synapses. Henceforth, the terms 'neurons' and 'synapses' will be used to
# describe the elements of an ANN. The behavior, in addition to the structure of an ANN, is also similar to biological
# neural networks: (1) each neuron holds a value which indicates its level of activation; (2) each directed edge
# (synapse) is assigned a value, known as its 'strength', which indicates how much influence the source neuron has on
# the target neuron; and (3) the activation a of a neuron i at the next time step is usually expressed as
# MISSING EXPRESSION
# where there are n neurons that connect to neuron i, aj is the activation of the jth neuron that connects to
# neuron i,wij is the weight of the synapse connecting neuron j to neuron i, and sigma() is a function that keeps the
# activation of neuron i from growing too large.

# In this project you will create an artificial neural network in Python, simulate its behavior over time, and visualize
# the resulting behavior.

########################################################################################################################

import math
import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.pyplot as neuronValuePlot
import numpy as np

from core01_TheHillClimber.Project_Hillclimber import MatrixCreate, MatrixRandomize

numNeurons = 10

# Project Details
# Tasks:
# 1. Back up your Python code from the previous project. Encapsulate your code from the previous project in a single
# file, such as Project_Hillclimber.py, such that when you run it you can reproduce all of the visualizations. This will
# prove to you that that code is working fine, as you will use it in this and subsequent projects.

# 2. Make a copy of Project_Hillclimber.py and call it Project_ANN.py. You will re-use the matrix functions you
# developed in that project here

# 3. First, we will create a neural network with 10 neurons. To do this, create a 50x10 matrix: element eij will store
# the value of neuron j at time step i. Name this matrix neuronValues.
neuronValues = MatrixCreate(50, 10)

# 4. Set each elements in the first row to a random value in [0, 1]: these values will represent the initial values of
# the neurons
MatrixRandomize(neuronValues[0])

# 5. To visualize the network, we will place the 10 neurons in a circular pattern as shown in Fig. 1a. Create a new
# matrix neuronPositions=MatrixCreate(2,10) which will store the two-dimensional position of each neuron such that
# neuronPositions[0,i] will store the x-value for neuron i and neuronPositions[1,i] will store the y-value for neuron i
neuronPositions = MatrixCreate(2, 10)

# 6. Now, compute the positions of the neurons as follows:
angle = 0.0
angleUpdate = 2 * math.pi / numNeurons

for i in xrange(0, numNeurons):
    x = math.sin(angle)
    y = math.cos(angle)
    angle = angle + angleUpdate
    neuronPositions[0][i] = x
    neuronPositions[1][i] = y

# 7. Now, use this matrix and the plot() function to create the visualization shown in Fig.1a. Hint: to create circles,
# you need to use plot(...,'ko',markerfacecolor=[1,1,1], markersize=18). Save this image
plt.plot(neuronPositions[0], neuronPositions[1], 'ko', markerfacecolor=[1, 1, 1], markersize=18)

# 8. To create the synapses, create a 10 x 10 matrix synapses and set each element to a value in [-1, 1], A synapse with
# a negative weight inhibits its target neuron: the stronger the activation of the originating neuron, the lower the
# activation of the target neuron.
synapses = MatrixCreate(10, 10)
for x in np.nditer(synapses, op_flags=['readwrite']):
    x[...] = random.uniform(-1, 1)


# 9. Create a plotting function that takes as input neuronPositions and draws a line between each pair of neurons. The
# resulting visualization should look like Fig. 1b. Save the resulting image. Note: If you want to draw a line between
# the two points (x1,y1) and (x2,y2), you can use plot([x1,x2],[y1,y2]), and not plot([x1,y1],[x2,y2]). Note: Each
# neuron has a self-connection: a synapse that connects it to itself. These synapses are not drawn in Fig. 1, but you
# can try to include them in the visualization if you like. (Thanks to /u/ismtrn for spotting this.)
def plotter(neuronPositions, synapses):
    for i in xrange(len(neuronPositions[0])):
        x1, y1 = neuronPositions[0][i], neuronPositions[1][i]
        for j in xrange(len(neuronPositions[0])):
            x2, y2 = neuronPositions[0][j], neuronPositions[1][j]

            if not x1 == x2 and not y1 == y2:  # avoid plotting self-loops
                # 10. Modify your plotting function such that it takes as input neuronPositions and synapses, and draws
                # gray lines (plot(...,color=[0.8,0.8,0.8])) for negatively-weighted synapses and black lines
                # (plot(...,color=[0,0,0])) for positively-weighted synapses. Save the resulting image, which should
                # look like Fig. 1c.
                global color
                color = None

                if synapses[i][j] < 0:
                    color = [0.8, 0.8, 0.8]  # gray - negative
                else:
                    color = [0, 0, 0]  # black - positive

                # 11. Modify your plotting function again such that the width of each line indicates the magnitude of
                # the corresponding synapse's weight. Note: The width of a line must be an integer value: e.g.,
                # plot(...,linewidth=2). To convert a synaptic weight to a number in 1,2,... use
                # w = int(10*abs(synapses[i,j]))+1, and then plot(...,linewidth=w). Save the resulting visualization,
                # which should look like Fig. 1d.
                w = int(10 * abs(synapses[i, j])) + 1

                plt.plot([x1, x2], [y1, y2], color=color, linewidth=w)

    plt.show()


plotter(neuronPositions, synapses)


# 12. Now create a function that updates each neuron in the network: neuronValues = Update (neuronValues,synapses,i).
# (You created neuronValues in step 3.) This function will compute the new values of all of the neurons and store them
# in row i of neuronValues. Thus you will have to call Update 49 times: one for row two, again for row three, and so on
# until row 50 is filled in. When called, this function will iterate through each element in row i of neuronValues, and
# for each such element j it will compute the sum
# MISSING EXPRESSION
# where ak is the value of the kth neuron on the previous row, and wjk is element ejk in the matrix synapses (in other
# words, it is the (w)eight of the synapse connecting neuron k to neuron j). If this temporary sum is less than zero,
# round it to zero; if it is larger than one, round it to one. Store the result-the new value of neuron j-in the correct
# place in neuronValues. Note: Before updating the neurons, make sure to set each neuron in the first row to a random
# value in [0,1]. Thing to think about: If the neurons are all set to zero initially, what do you think their values
# will be in subsequent time steps?
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


for i in xrange(1, 50):
    neuronValues = Update(neuronValues, synapses, i)

# np.set_printoptions(threshold=np.nan)
# np.set_printoptions(linewidth=500)
# print neuronValues

# 13. Now use the matrix imaging function you developed in the previous project to visualize how the neuron values
# change over time. This should produce an image similar to that shown in Fig. 1e. Save this image. The image does not
# need to look exactly like that of Fig. 1e. In fact, re-run your program several times, and compare the images
# produced. Notice that the patterns of neuron activation vary greatly from one run to the next. Why do you think this
# is so?
neuronValuePlot.clf()
neuronValuePlot.imshow(neuronValues, cmap=cm.gray, aspect='auto', interpolation='nearest')
neuronValuePlot.show()
