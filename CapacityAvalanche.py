import numpy as np 

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons 
        self.weights = np.zeros((num_neurons, num_neurons), dtype=np.float64)  # initialize weight matrix with zeros and set dtype to float64

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # sigmoid activation function

    def train(self, patterns, learning_rate=0.1, epochs=100, alpha=0.1):  # increased learning rate and epochs
        self.weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float64)  # reset weight matrix to zero with float64

        for epoch in range(epochs):  # gradient descent loop
            print(f"Epoch {epoch+1}/{epochs}")
            gradient = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float64)

            for p in patterns:  # iterate over each pattern
                t = (p + 1) / 2  # convert pattern to 0/1
                print(f"Pattern: {p}")
                for i in range(self.num_neurons):
                    a_i = np.dot(self.weights[i], p)  # calculate activation for neuron i
                    y_i = self.sigmoid(a_i)  # calculate output for neuron i
                    error = t[i] - y_i  # calculate error
                    delta_w = learning_rate * error * p  # calculate weight update

                    print(f"Neuron {i}: Activation: {a_i}, Output: {y_i}, Error: {error}, Delta W: {delta_w}") # debugging line

                    # Update weights symmetrically
                    for j in range(self.num_neurons):
                        if i != j:
                            self.weights[i][j] += delta_w[j]
                            self.weights[j][i] += delta_w[j]  # ensure symmetry

            np.fill_diagonal(self.weights, 0)  # set diagonal elements to zero to avoid self-connections
            print(f"Weight matrix after epoch {epoch+1}:\n{self.weights}\n")

    def recall(self, pattern, steps=10):
        for step in range(steps):  # iterate for a fixed number of steps
            new_pattern = pattern.copy()  # create a copy of the current pattern
            for i in range(self.num_neurons):  # iterate over each neuron
                raw_input = np.dot(self.weights[i], pattern)  # calculate the weighted sum of inputs for neuron i
                new_pattern[i] = 1 if raw_input >= 0 else -1  # update the state of neuron i
            pattern = new_pattern.copy()  # update the pattern for the next step
            print(f"Pattern after step {step + 1}: {pattern}")  # print the pattern after this step
        return pattern  # return the recalled pattern

# define patterns
patterns = np.array([[1, -1, 1, -1], [-1, -1, 1, 1]], dtype=np.float64)

# initialize and train the network
hopfield = HopfieldNetwork(num_neurons=4)
hopfield.train(patterns, learning_rate=0.1, epochs=10)  # limited epochs for debugging

# print the weight matrix
print("Final weight matrix after training:\n", hopfield.weights)

# test the network with a corrupted pattern
original_pattern = np.array([1, -1, 1, -1], dtype=np.float64)  # define the original pattern
corrupted_pattern = np.array([1, -1, 1, 1], dtype=np.float64)  # define a corrupted version of the original pattern
recalled_pattern = hopfield.recall(corrupted_pattern.copy())  # recall the pattern from the corrupted version

print("Original pattern:   ", original_pattern)  # print the original pattern
print("Corrupted pattern:  ", corrupted_pattern)  # print the corrupted pattern
print("Recalled pattern:   ", recalled_pattern)  # print the recalled pattern
