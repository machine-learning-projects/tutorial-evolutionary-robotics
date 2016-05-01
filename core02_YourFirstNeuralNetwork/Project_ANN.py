# core02 - Your First Neural Network - https://www.uvm.edu/~ludobots/index.php/ER/Assignment2

import matplotlib.pyplot as plt


def plotter(neuronPositions, synapses):
    for i in xrange(len(neuronPositions[0])):
        x1, y1 = neuronPositions[0][i], neuronPositions[1][i]
        for j in xrange(len(neuronPositions[0])):
            x2, y2 = neuronPositions[0][j], neuronPositions[1][j]

            if not x1 == x2 and not y1 == y2:  # avoid plotting self-loops
                global color
                color = None

                if synapses[i][j] < 0:
                    color = [0.8, 0.8, 0.8]  # gray - negative
                else:
                    color = [0, 0, 0]  # black - positive

                w = int(10 * abs(synapses[i, j])) + 1

                plt.plot([x1, x2], [y1, y2], color=color, linewidth=w)

    plt.show()


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
