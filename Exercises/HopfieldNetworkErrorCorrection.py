# Exercise 42.5 in the book
import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons  # number of neurons in the network
        self.weights = np.zeros((num_neurons, num_neurons))  # initialize the weight matrix with zeros

    def train(self, patterns):
        self.weights = np.zeros((self.num_neurons, self.num_neurons))  # reset the weight matrix to zero
        for p in patterns:  # iterate over each pattern
            self.weights += np.outer(p, p)  # update the weights using the outer product of the pattern with itself
        np.fill_diagonal(self.weights, 0)  # diagonal elements set to zero to avoid self-connections

    # asynchronous updates: updates each neuron one by one each 'step'
    def recall(self, pattern, steps=10):
        for step in range(steps):  # iterate 'steps' amount
            for i in range(self.num_neurons):  # iterate over each neuron
                raw_input = np.dot(self.weights[i], pattern)  # calculate the weighted sum of inputs for neuron i
                pattern[i] = 1 if raw_input >= 0 else -1  # update the state of neuron i (Binary Threshold Activation Function)
            print(f"Pattern after step {step + 1}: {pattern}")  # print pattern after each step for debugging
        return pattern  # return the recalled pattern

    # new code modification for DJCM visual setup
    def visualize_pattern(self, pattern, title):
        plt.imshow(pattern.reshape(5, 5), cmap='binary')  # reshape the pattern to a 5x5 grid and display it
        plt.title(title)  # set the title of the plot
        plt.axis('off')  # turn off the axis

# Define patterns for D, J, C, M (5x5 grid representation)
patterns = np.array([
    [1, 1, 1, -1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, -1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, -1, -1],  # pattern for 'D'

    [1, 1, 1, 1, 1,
     -1, -1, -1, 1, -1,
     -1, -1, -1, 1, -1,
     1, -1, -1, 1, -1,
     -1, 1, 1, 1, -1],  # pattern for 'J'

    [1, 1, 1, 1, 1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, -1, -1, -1, -1,
     1, 1, 1, 1, 1],  # pattern for 'C'

    [1, -1, -1, -1, 1,
     1, -1, -1, 1, -1,
     1, -1, 1, -1, -1,
     1, 1, -1, -1, -1,
     1, -1, -1, -1, 1]   # pattern for 'M'
])

# Initialize and train the network
hopfield = HopfieldNetwork(num_neurons=25)  # create a Hopfield network with 25 neurons
hopfield.train(patterns)  # train the network with the given patterns

# visualize rhe original patterns
plt.figure(figsize=(10, 8))  # set the size of the figure
for i, pattern in enumerate(patterns):  # iterate over each pattern
    plt.subplot(2, 4, i + 1)  # create a subplot for each pattern
    hopfield.visualize_pattern(pattern, f'Original Pattern {i + 1}')  # visualize each pattern
plt.show()  # display the figure

# Define corrupted patterns for testing
corrupted_patterns = [
    np.array([1, 1, 1, -1, -1,
              1, -1, -1, -1, -1,
              1, 1, -1, -1, -1,
              1, -1, -1, -1, -1,
              1, 1, 1, -1, -1]),  # corrupted version of 'D'

    np.array([1, 1, 1, 1, 1,
              -1, -1, -1, 1, -1,
              -1, -1, -1, 1, -1,
              1, -1, -1, 1, -1,
              -1, 1, 1, 1, -1]),  # corrupted version of 'J'

    np.array([1, 1, 1, 1, 1,
              1, -1, -1, -1, -1,
              1, -1, -1, -1, -1,
              1, -1, -1, -1, -1,
              1, 1, 1, 1, -1]),  # corrupted version of 'C'

    np.array([1, -1, -1, -1, 1,
              1, -1, -1, 1, -1,
              1, -1, 1, -1, -1,
              1, 1, -1, -1, -1,
              1, -1, -1, -1, -1])  # corrupted version of 'M'
]

# visualize recalled patterns after testing with corrupted patterns
plt.figure(figsize=(10, 8))  # set the size of the figure
for i, pattern in enumerate(corrupted_patterns):  # iterate over each corrupted pattern
    recalled_pattern = hopfield.recall(pattern.copy())  # recall the pattern from the corrupted version
    plt.subplot(2, 4, i + 5)  # create a subplot for each recalled pattern
    hopfield.visualize_pattern(recalled_pattern, f'Recalled Pattern {i + 1}')  # visualize each recalled pattern
plt.show()  # display the figure
