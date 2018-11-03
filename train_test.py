import csv
import numpy as np
from layers import *
from backpropagate import *
from neural_network import *
import matplotlib.pyplot as plt
import sys

############# Flags. ##############
batch_size = 512
epoch = 500

lr = 1e-3
scale = 1.0

modes = ["third", "second", "first"]

dir = "../../Question_2_1/"
train_data_file = dir + "x_train.csv"
train_labels_file = dir + "y_train.csv"
test_data_file = dir + "x_test.csv"
test_labels_file = dir + "y_test.csv"
####################################

# Plot graphs
def plot_accuracy(mode, data, num_iterations, batch_size, lr, scale):
    plt.plot(data[0], 'r', label='14-14x28-4')
    plt.plot(data[1], 'b', label='14-28x6-4')
    plt.plot(data[2], 'g', label='14-100-40-4')
    plt.legend(loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.suptitle('Accuracy against Iterations', fontsize=14)
    plt.title("batch_size " + str(batch_size) + " lr " + str(lr) + " scale " + str(scale), fontsize=8)
    plt.savefig("acc_" + mode + "_" + str(num_iterations) + "_" + str(batch_size) + "_" + str(lr) + "_" + str(scale) + ".png")
    plt.show()

def plot_loss(mode, data, num_iterations, batch_size, lr, scale):
    plt.plot(data[0], 'r', label='14-14x28-4')
    plt.plot(data[1], 'b', label='14-28x6-4')
    plt.plot(data[2], 'g', label='14-100-40-4')
    plt.legend(loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.suptitle('Loss against Iterations', fontsize=14)
    plt.title("batch_size " + str(batch_size) + " lr " + str(lr) + " scale " + str(scale), fontsize=8)
    plt.savefig("loss_" + mode + "_" + str(num_iterations) + "_" + str(batch_size) + "_" + str(lr) + "_" + str(scale) + ".png")
    plt.show()

# Split data into batches
def split_to_batches(data, labels):
    split = len(data) / batch_size
    batch_data = np.array_split(data, split)
    batch_label = np.array_split(labels, split)
    return batch_data, batch_label

# Shuffle data + labels
def shuffle_data(data, labels):
    p = np.random.permutation(len(data))
    return data[p], labels[p]

# Build the NeuralNetwork and return
def setup_network(input_num, output_num, lr, scale):
    if mode is "first":
        return first(input_num, output_num, lr=lr, scale=scale)
    elif mode is "second":
        return second(input_num, output_num, lr=lr, scale=scale)
    else:
        return third(input_num, output_num, lr=lr, scale=scale)

# Normalize training data
def normalize(train_data):
    for i in range(0, len(train_data)):
        mean = np.mean(train_data[i])
        stdev = np.std(train_data[i])
        np.subtract(train_data[i], mean)
        if stdev > 0:
            np.divide(train_data[i], stdev)
    return train_data

# Convert integer labels to binary
def get_binary_labels(raw_matrix):
    labels = np.zeros(shape=(len(raw_matrix), 4), dtype=float)
    for i in range(0, len(raw_matrix)):
        labels[i][int(raw_matrix[i][0])] = 1
    return labels

# Helper function to read in data files
def read_data(f, is_label):
    csv_reader = csv.reader(open(f, "r"), delimiter=",")
    raw_matrix = list(csv_reader)
    if is_label:
        return get_binary_labels(raw_matrix)
    return np.array(raw_matrix, dtype=float)

# Capture stats per iteration
train_losses = []
train_acc = []
test_losses = []
test_acc = []

for mode in modes:
    # Read train and test data
    train_data = read_data(train_data_file, False)
    train_labels = read_data(train_labels_file, True)
    test_data = read_data(test_data_file, False)
    test_labels = read_data(test_labels_file, True)

    # Normalize data
    # train_data = normalize(train_data)

    # Setup NN architecture
    network = setup_network(14, 4, lr, scale)

    num_iterations = 0

    mode_train_losses = []
    mode_test_losses = []
    mode_train_acc = []
    mode_test_acc = []

    for j in range(0, epoch):
        # Shuffle train data and create batches
        p_train_data, p_train_labels = shuffle_data(train_data, train_labels)
        batch_data, batch_label = split_to_batches(p_train_data, p_train_labels)

        for i in range(0, len(batch_data)):
            # Train pass
            train_accuracy, train_loss = network_forward(network, batch_data[i], batch_label[i])
            network_backward(network)
            print(mode, ",", j, ",", i, ",", num_iterations, "TRAIN", "loss", train_loss, "accuracy", train_accuracy)

            # Test pass
            test_accuracy, test_loss = network_forward(network, test_data, test_labels)
            print(mode, ",", j, ",", i, ",", num_iterations, "TEST", "loss", test_loss, "accuracy", test_accuracy)

            # Take down stats for plotting
            mode_train_losses.append(train_loss)
            mode_train_acc.append(train_accuracy)
            mode_test_losses.append(test_loss)
            mode_test_acc.append(test_accuracy)

            num_iterations += 1
                
        # Momentum decay
        # for layer in network:
        #     if type(layer) is FullyConnectedLayer:
                # layer.lr *= 0.9
                # layer.scale *= 0.9
    
    train_losses.append(mode_train_losses)
    train_acc.append(mode_train_acc)
    test_losses.append(mode_test_losses)
    test_acc.append(mode_test_acc)

# Plot
plot_loss("train", train_losses, num_iterations, batch_size, lr, scale)
plot_loss("test", test_losses, num_iterations, batch_size, lr, scale)
plot_accuracy("train", train_acc, num_iterations, batch_size, lr, scale)
plot_accuracy("test", test_acc, num_iterations, batch_size, lr, scale)