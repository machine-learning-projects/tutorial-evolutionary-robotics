# tutorial-evolutionary-robotics

[Ludobots - Education in Evolutionary Robotics](https://www.uvm.edu/~ludobots/index.php/Main/ER2014)

[Ludobots subreddit](https://www.reddit.com/r/ludobots) and [Index](https://www.reddit.com/r/ludobots/wiki/index#welcome)
[Assignments](https://www.reddit.com/r/ludobots/wiki/tree#ludobotstreeurl)

## core01 - [The Hill Climber](https://www.uvm.edu/~ludobots/index.php/ER/Assignment1)
In this project, rather than evolving robots, you will simply evolve vectors of the form v = e1, . . . en, where the ith element in the vector may take on a real number in [0, 1]. The fitness of a vector v, denoted f(n) we will define as the mean value of all of its elements:

Thus, the hill climber will search for vectors in which all of the elements have values near one. In a later project, you will re-use your hill climber to evolve artificial neural networks; in another project, you will use it to evolve robots.

## core02 - [Your First Neural Network](https://www.uvm.edu/~ludobots/index.php/ER/Assignment2)
In this project you will be creating an artificial neural network (ANN). There are many kinds of ANNs, but they all share one thing in common: they are represented as a directed graph in which the nodes are models of biological neurons, and edges are models of biological synapses. Henceforth, the terms ‘neurons’ and ‘synapses’ will be used to describe the elements of an ANN. The behavior, in addition to the structure of an ANN, is also similar to biological neural networks: (1) each neuron holds a value which indicates its level of activation; (2) each directed edge (synapse) is assigned a value, known as its ‘strength’, which indicates how much influence the source neuron has on the target neuron; and (3) the activation a of a neuron i at the next time step is usually expressed as


where there are n neurons that connect to neuron i, aj is the activation of the jth neuron that connects to neuron i, wij is the weight of the synapse connecting neuron j to neuron i, and σ() is a function that keeps the activation of neuron i from growing too large.

In this project you will create an artificial neural network in Python, simulate its behavior over time, and visualize the resulting behavior.

## core03 - [Evolving a Neural Network](https://www.uvm.edu/~ludobots/index.php/ER/Assignment3)
In this project you will apply the hillclimber you developed in this project to the artificial neural networks (ANNs) you developed in this project. Below is a summary of what you will implement in this project.

1. Create a (numNeurons=)10 × (numNeurons=)10 matrix called parent to hold the synapse weights for an ANN with numNeurons=10 neurons.
2. Randomize the parent synapse matrix.
3. Create a (numUpdates=)10 × (numNeurons=)10 matrix called neuronValues to hold the values of the neurons as the network is updated. The first row stores the initial values of the neurons, which for this project will all initially be set to 0.5. The second row will store the new values of each neuron, and so on.
4. Create a vector of length 10 called desiredNeuronValues that holds the values that each neuron in the ANN should reach. We’ll compare the final values of the neurons—the last row of neuronValues—to this vector. The closer the match, the higher the fitness of the synapse matrix.
5. Update neuronValues nine times (thus filling in rows two through 10) using the parent synapse weights.
6. The program will then loop through 1000 generations. For each loop: (1) Create the child synapse matrix by copying and perturbing the parent synapse matrix. (2) Set each element in the first row of neuronValues to 0.5. (3) Update the neuronValues of the ANN nine times using the child synapse values. (4) Calculate and save the fitness of the child synapses as childFitness. (5) If the childFitness is better than parentFitness, replace the parent synapse matrix with the child synapse matrix and set parentFitness to the childFitness value.
