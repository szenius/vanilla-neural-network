import layers
import numpy as np

def network_forward(network, input_data, label_data):
    for layer in network:
        if type(layer) is not layers.SoftmaxOutput_CrossEntropyLossLayer:
            input_data = layer.forward(input_data)
        else:
            return layer.eval(input_data, label_data)

def network_backward(network):
    for layer in reversed(network):
        if type(layer) is layers.SoftmaxOutput_CrossEntropyLossLayer:
            gradient = layer.backward()
        else:
            gradient = layer.backward(gradient)