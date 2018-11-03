from layers import *

def debug(input_num, output_num):
    return [
        FullyConnectedLayer(input_num, 6, w=np.ones(shape=(input_num, 6))),
        ReLULayer(),
        FullyConnectedLayer(6, output_num, w=np.ones(shape=(6, output_num))),
        SoftmaxOutput_CrossEntropyLossLayer()
    ]

def first(input_num, output_num, w=None, b=None, lr=1e-3, scale=1.0):
    if w is None and b is None:
        return [
            FullyConnectedLayer(input_num, 100, lr=lr, scale=scale),
            ReLULayer(),
            FullyConnectedLayer(100, 40, lr=lr, scale=scale),
            ReLULayer(),
            FullyConnectedLayer(40, output_num, lr=lr, scale=scale),
            SoftmaxOutput_CrossEntropyLossLayer()
        ]
    else:
        return [
            FullyConnectedLayer(input_num, 100, w[0], b[0], lr=lr, scale=scale),
            ReLULayer(),
            FullyConnectedLayer(100, 40, w[1], b[1], lr=lr, scale=scale),
            ReLULayer(),
            FullyConnectedLayer(40, output_num, w[2], b[2], lr=lr, scale=scale),
            SoftmaxOutput_CrossEntropyLossLayer()
        ]

def second(input_num, output_num, w=None, b=None, lr=1e-3, scale=1.0):    
    network = list()

    if w is None and b is None:
        network.append(FullyConnectedLayer(input_num, 28, lr=lr, scale=scale))
        network.append(ReLULayer())
        for i in range(0, 5):
            network.append(FullyConnectedLayer(28, 28, lr=lr, scale=scale))
            network.append(ReLULayer())
        network.append(FullyConnectedLayer(28, output_num, lr=lr, scale=scale))
        network.append(SoftmaxOutput_CrossEntropyLossLayer())
    else:
        network.append(FullyConnectedLayer(input_num, 28, w[0], b[0], lr=lr, scale=scale))
        network.append(ReLULayer())
        for i in range(0, 5):
            network.append(FullyConnectedLayer(28, 28, w[i + 1], b[i + 1], lr=lr, scale=scale))
            network.append(ReLULayer())
        network.append(FullyConnectedLayer(28, output_num, w[6], b[6], lr=lr, scale=scale))
        network.append(SoftmaxOutput_CrossEntropyLossLayer())
    return network

def third(input_num, output_num, w=None, b=None, lr=1e-3, scale=1.0):
    network = list()

    if w is None and b is None:
        network.append(FullyConnectedLayer(input_num, 14, lr=lr, scale=scale))
        network.append(ReLULayer())
        for i in range(0, 27):
            network.append(FullyConnectedLayer(14, 14, lr=lr, scale=scale))
            network.append(ReLULayer())
        network.append(FullyConnectedLayer(14, output_num, lr=lr, scale=scale))
        network.append(SoftmaxOutput_CrossEntropyLossLayer())
    else:
        network.append(FullyConnectedLayer(input_num, 14, w[0], b[0], lr=lr, scale=scale))
        network.append(ReLULayer())
        for i in range(0, 27):
            network.append(FullyConnectedLayer(14, 14, w[i + 1], b[i + 1], lr=lr, scale=scale))
            network.append(ReLULayer())
        network.append(FullyConnectedLayer(14, output_num, w[28], b[28], lr=lr, scale=scale))
        network.append(SoftmaxOutput_CrossEntropyLossLayer())
    return network

