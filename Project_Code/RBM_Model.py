import numpy as np

class RBM:
    def __init__(self, visible_units, hidden_units):
        # initialize the visible and hidden units
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        
        # here I initialize weights with small random values
        self.weights = np.random.randn(hidden_units, visible_units) * 0.1
        
        # initialize hidden and visible biases with zeros
        self.h_bias = np.zeros(hidden_units)
        self.v_bias = np.zeros(visible_units)

    def sample_hidden(self, v):
        # compute the hidden unit probabilities
        h_prob = self.sigmoid(np.dot(self.weights, v) + self.h_bias[:, np.newaxis])
        
        # sample hidden units based on probabilities
        h_sample = (h_prob > np.random.rand(self.hidden_units, v.shape[1])).astype(int)
        
        # return hidden probabilities and samples
        return h_prob, h_sample

    def sample_visible(self, h):
        # compute the visible unit probabilities
        v_prob = self.sigmoid(np.dot(self.weights.T, h) + self.v_bias[:, np.newaxis])
        
        # sample visible units based on probabilities
        v_sample = (v_prob > np.random.rand(self.visible_units, h.shape[1])).astype(int)
        
        # return visible probabilities and samples
        return v_prob, v_sample

    def sigmoid(self, x):
        # sigmoid function as defined in Chapter 42
        return 1 / (1 + np.exp(-x))

    def hebbian_update(self, v, h_prob, lr=0.1):
        # update weights using Hebbian learning rule
        self.weights += lr * (np.dot(h_prob, v.T) - np.dot(h_prob, v.T)) / v.shape[1]
        
        # update hidden biases
        self.h_bias += lr * np.mean(h_prob - h_prob, axis=1)
        
        # update visible biases
        self.v_bias += lr * np.mean(v - v, axis=0)

    def train(self, data, epochs=10, lr=0.1):
        # iterate over the number of epochs
        for epoch in range(epochs):
            # iterate over each data sample
            for v in data:
                v = v.reshape(-1, 1)  # ensure v is a column vector
                
                # sample hidden units
                h_prob, h_sample = self.sample_hidden(v)
                
                # update weights and biases using Hebbian learning
                self.hebbian_update(v, h_prob, lr)
            
            # print progress for each epoch
            print(f'Epoch {epoch + 1}/{epochs} complete')

    def recall(self, data):
        # sample hidden units from input data
        h_prob, h_sample = self.sample_hidden(data.T)
        
        # recall visible units from hidden units
        v_prob, v_sample = self.sample_visible(h_prob)
        
        # return the recall visible units
        return v_prob.T

    def add_noise(self, data, corruption_level=0.3):
        # create a copy of the data
        corrupted_data = data.copy()
        
        # create a mask to corrupt data
        mask = np.random.binomial(1, corruption_level, data.shape)
        
        # apply the mask to corrupt the data
        corrupted_data[mask == 1] = 0
        
        # return the corrupted data
        return corrupted_data

    def error_correction(self, corrupted_data):
        # recall data from corrupted input
        recalled_data = self.recall(corrupted_data)
        
        # apply error correction to the recalled data
        corrected_data = (recalled_data > 0.5).astype(int)
        
        # return the corrected data
        return corrected_data