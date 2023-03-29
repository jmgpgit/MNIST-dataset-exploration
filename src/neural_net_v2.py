import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from more_itertools import chunked
import pickle

def display_image(img, img_size=28):
    plt.imshow(np.array(img).reshape(img_size,img_size), cmap=plt.cm.binary)
    plt.gcf().axes[0].set_axis_off()
    plt.show()
   
    
def activation(x, d=False, fn = 'sigmoid'):
    """Activation function for the neural network.
    Args:
        x (np.array): Input to the activation function.
        d (bool): If True, return the derivative of the activation function.
        fn (str): The activation function to use. Options are 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'softmax', and 'linear'.
    
    Returns:
        np.array: The output of the activation function.
    """
    if fn == 'sigmoid':
        if d:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))
    if fn == 'tanh':
        if d:
            return 1 - np.tanh(x)**2
        return np.tanh(x)
    if fn == 'relu':
        if d:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)
    if fn == 'leaky_relu':
        if d:
            return np.where(x > 0, 1, 0.01)
        return np.maximum(0.01*x, x)
    if fn == 'softmax':
        if d:
            return x * (1 - x)
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    if fn == 'linear':
        if d:
            return 1
        return x
    raise ValueError("Invalid activation function: %s" % fn)

class Neural_Network:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_1 = np.random.randn(self.input_size, self.hidden_size)
        self.weights_2 = np.random.randn(self.hidden_size, self.output_size)
        self.bias_1 = np.zeros(hidden_size)
        self.bias_2 = np.zeros(output_size)
        
    def feed_forward(self, input, fn = 'sigmoid'):
        hidden_layer = activation(input @ self.weights_1 + self.bias_1, fn= fn)
        output_layer = activation(hidden_layer @ self.weights_2 + self.bias_2, fn=fn)
        return output_layer , hidden_layer
    
    def back_prop(self, input, target, learning_rate = 0.1, fn = 'sigmoid'):
        output_layer, hidden_layer = self.feed_forward(input, fn)
        
        output_error = output_layer - target
        hidden_error = output_error @ self.weights_2.T
        
        output_gradient = output_error * activation(output_layer, d=True, fn=fn)
        hidden_gradient = hidden_error * activation(hidden_layer, d=True, fn=fn)
        
        self.weights_2 -= learning_rate * np.outer(hidden_layer.T, output_gradient)
        self.weights_1 -= learning_rate * np.outer(input.T, hidden_gradient)
        
        self.bias_2 -= learning_rate * output_gradient
        self.bias_1 -= learning_rate * hidden_gradient
        
    def train(self, inputs, targets, epochs = 10, learning_rate = 0.1, fn = 'sigmoid'):
        training_data = list(zip(inputs, targets))
        for _ in trange(epochs):
            np.random.shuffle(training_data)
            for input, target in training_data:
                self.back_prop(input, target, learning_rate, fn)
        
    def predict(self, input, fn = 'sigmoid'):
        output, _ = self.feed_forward(input, fn)
        return np.argmax(output), output
    
    def test(self, inputs, targets, fn = 'sigmoid'):
        correct = 0
        for input, target in zip(inputs, targets):
            prediction, _ = self.predict(input, fn)
            if prediction == np.argmax(target):
                correct += 1
        return correct / len(inputs)
        