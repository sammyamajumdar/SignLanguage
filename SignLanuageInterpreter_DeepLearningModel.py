import tensorflow  as tf
import numpy as np
import h5py
import math
from python_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

# Load the dataset in their respective groups
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
#print(X_train_orig.shape)
# Flatten the images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T    
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T  

#normalize the images
X_train = X_train_flatten/255
X_test = X_test_flatten/255

# Convert training and testing labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

#Check dimensions of the test and training sets
"""print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))"""

# Create placeholders to feed data during training

def create_placeholders(n_x, n_y):
    # n_x = size of an image vector, 64*64*3 = 12288
    # n_y = number of classes = 6

    X = tf.placeholder(tf.float32, [n_x, None], name = "X" )
    Y = tf.placeholder(tf.float32, [n_y, None], name = "Y")

    return X, Y

def initialize_parameters():
    """the shapes are :
        W1 = (25, 12288)  
        b1 = (25, 1)
        W2 = (12, 25)
        b1 = (12, 1)
        W3 = (6, 12)
        b1 = (6, 1)
    this function returns a dictionary containing tensors W1,b1,W2,b2,W3,b3
    Weights are Xavier initialized and biases are zero initialized"""

    W1 = tf.get_variable("W1", [25, 12288], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [25, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [12, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [6, 1], initializer = tf.zeros_initializer())

    params = {"W1" : W1, "b1" : b1, "W2" : W2, "b2" : b2, "W3" : W3, "b3" : b3}

    return params
# check the function initialize parameters
"""tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))"""

def forward_propagation(X, params):
    #LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]

    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3

    return Z3
# Check the forward propagation function
"""tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))"""

def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    return cost

#check compute_cost function
"""tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))"""

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost=True):
    (n_x, m) = X_train.shape
    tf.set_random_seed(1)
    seed = 3
    ops.reset_default_graph()
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Training Loop
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatch = int(m/minibatch_size)
            seed += 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / minibatch_size

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        

        # Plot learning curve
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print("Parameters have been successfully trained.")
        # Calculate correct predictioms
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters

                

parameters = model(X_train, Y_train, X_test, Y_test)