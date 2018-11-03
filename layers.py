import numpy as np
from numpy import nan

class FullyConnectedLayer:
    def __init__(self, num_input, num_output, w=None, b=None, lr=1e-1, scale=1):
        # Initialize weights
        if w is None:
            w = np.random.randn(num_input, num_output) * np.sqrt(2.0/(num_input))
        self.w = w

        # Initialize bias
        if b is None:
            b = np.random.randn(1, num_output) * np.sqrt(2.0/(num_output))
        self.b = b

        self.lr = lr
        self.scale = scale
        
    # Backpropagation forward pass
    def forward(self, input_data):
        self.x = input_data
        return np.add(np.dot(self.x, self.w), self.b)

    # Backpropagation backward pass
    def backward(self, gradient_data): 
        original_w = np.copy(self.w)

        # update bias
        self.db = np.array([np.mean(gradient_data, axis=0)])
        self.b = np.subtract(np.dot(self.scale, self.b), np.dot(self.lr, self.db))

        # update weight
        self.dw = np.dot(gradient_data.T, self.x).T / len(self.x)
        self.w = np.subtract(np.dot(self.scale, self.w), np.dot(self.lr, self.dw))

        return np.dot(gradient_data, np.transpose(original_w))

class ReLULayer:
    def __init__(self):
        pass

    # Backpropagation forward pass
    def forward(self, input_data):
        self.x = input_data
        return input_data * (input_data > 0)

    # Backpropagation backward pass
    def backward(self, gradient_data):
        return gradient_data * (self.x > 0)

class SoftmaxOutput_CrossEntropyLossLayer():
    def __init__(self):
        pass
    
    # Compute stable softmax, i.e. transform predictions so that they add up to 1
    def stable_softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.array([np.sum(exps, axis=1)]).T

    # Compute cross entropy loss
    def compute_cross_entropy(self, y, epsilon=1e-50):
        m = y.shape[0]
        p_y = np.copy(self.p)
        for i in range(0, len(self.p)):
            p_y[i] = -np.log(p_y[i] + epsilon) * y[i]
        return np.sum(p_y) / m

    # Compute accuracy
    def compute_accuracy(self, y):
        correct_predictions = 0
        for i in range(0, y.shape[0]):
            if self.is_correct_prediction(self.p[i], y[i]):
                correct_predictions += 1
        return correct_predictions / len(y)
    
    # Returns True if the prediction was correct, False otherwise
    def is_correct_prediction(self, p, y):
        return np.argmax(p) == np.argmax(y)

    # Compute loss and accuracy
    def eval(self, input_data, label_data):
        self.y = label_data
        self.p = self.stable_softmax(input_data)
        loss = self.compute_cross_entropy(label_data)
        accuracy = self.compute_accuracy(label_data)
        return accuracy, loss

    # Backpropagation backward pass
    def backward(self):
        return np.subtract(self.p, self.y)