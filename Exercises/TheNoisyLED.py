# Exercise 39.5 in the book
import numpy as np 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # sigmoid activation function

def softmax(a):
    exp_a = np.exp(a - np.max(a))  # subtract max for numerical stability
    return exp_a / exp_a.sum()  # calculate softmax probabilities

class NoisyLED:
    def __init__(self, f=0.1):
        self.f = f  # probability of flipping the state
        self.log_ratio = np.log((1 - f) / f)  # compute log ratio
        # define the character states for 2 and 3
        self.characters = {
            2: np.array([1, 1, 0, 1, 1, 0, 1]),
            3: np.array([1, 1, 1, 1, 0, 0, 1])
        }
    
    def calc_weights(self):
        self.weights = {}  # initialize weights dictionary
        for s, c_s in self.characters.items():  # loop over each character
            self.weights[s] = self.log_ratio * (2 * c_s - 1)  # compute weights for character

    def prob_s_given_x(self, x, s):
        w = self.weights[s]  # get the weights for character s
        theta = 0  # set theta to 0 assuming equal priors
        return sigmoid(np.dot(w, x) + theta)  # compute probability using sigmoid

    def prob_multichar(self, x):
        # calculate activation for each character
        a_s = {s: np.dot(self.weights[s], x) for s in self.characters}
        a = np.array(list(a_s.values()))  # convert activations to array
        return softmax(a)  # compute softmax probabilities

led = NoisyLED(f=0.1)  # instance with f=0.1
led.calc_weights()  # compute the weights for the characters

# given state vector x for a noisy display
x = np.array([1, 1, 0, 1, 1, -1, 1])  # example noisy state vector

# compute probability P(s=2|x)
p_s2_given_x = led.prob_s_given_x(x, 2)
print(f"P(s=2|x) = {p_s2_given_x}")  # prints the probability

# compute probabilities for multiple characters
p_multiclass = led.prob_multichar(x)
print(f"P(s|x) for multiple characters: {p_multiclass}")  # print the probabilities
