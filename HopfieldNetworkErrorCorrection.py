# this program will use a Hopfield Network to do error-correction

import numpy as np 

class HopfieldNetwork: # class declaration
    def __init__(self, num_neurons): # constructor that initializes an instance's attributes (Hopfield Network class instance)
        self.num_neurons = num_neurons  # number of neurons in the network (this is an attribute)
        self.weights = np.zeros((num_neurons, num_neurons))  # initialize the weight matrix with zeros (this is an attribute)

    def train(self, patterns): # function declaration
        self.weights = np.zeros((self.num_neurons, self.num_neurons))  # reset the weight matrix to zero
        for p in patterns:  # tterate over each pattern
            self.weights += np.outer(p, p)  # update the weights using the outer product of the pattern with itself
        np.fill_diagonal(self.weights, 0)  # diagonal elements set to zero to avoid self-connections

    def recall(self, pattern, steps=10): # another function declaratio
        for step in range(steps):  # iterate 'steps' amount
            new_pattern = pattern.copy()  # make copy of the current pattern
            for i in range(self.num_neurons):  # iterate over each neuron
                raw_input = np.dot(self.weights[i], pattern)  # calculate the weighted sum of inputs for neuron i
                new_pattern[i] = 1 if raw_input >= 0 else -1  # update the state of neuron i (literally a Binary Threshold Activation Function)
            pattern = new_pattern.copy()  # update the pattern for the next step
            print(f"Pattern after step {step + 1}: {pattern}")  # this is just for debugging. Prints pattern after each step
        return pattern  # this function needs to return the recalled pattern so i did that here

# Define patterns
patterns = np.array([[1, -1, 1, -1]])  # here i define a simple training pattern

# Initialize and train the network
hopfield = HopfieldNetwork(num_neurons=4)  # here i create a Hopfield network with 4 neurons
hopfield.train(patterns)  # then you train the network with the given patterns

# Print the weight matrix
print("Weight matrix after training:\n", hopfield.weights) 

# Test the network with a corrupted pattern
original_pattern = np.array([1, -1, 1, -1])  # here I define the original pattern (in this case the same as the training pattern)
corrupted_pattern = np.array([1, -1, 1, 1])  # this is a corrupted version of the original pattern to test against for error-correction
recalled_pattern = hopfield.recall(corrupted_pattern.copy())  # recall the pattern from the corrupted version (recalled should be corrected)

print("Original pattern:   ", original_pattern)  # the original pattern
print("Corrupted pattern:  ", corrupted_pattern)  # the corrupted pattern
print("Recalled pattern:   ", recalled_pattern)  # the recalled pattern
