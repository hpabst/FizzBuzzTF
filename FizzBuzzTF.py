import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math



def binary_encode(i):
    """
    Encodes 32 bit integer into binary.
    :param i: Integer to encode.
    :return: Binary representation of integer.
    """
    return np.array([i >> j & 1 for j in range(32)])

def integer_encode(b):
    """
    Decodes a binary array back into an integer.
    :param b:
    :return:
    """
    num = 0
    for i in range(len(b)):
        num += math.pow(2, i) * b[i]
    return num

def fizzbuzz_encode(i):
    """
    Encodes number as expected fizzbuzz output.
    :param i:
    :return:
    """
    if i%15 == 0: return np.array([0, 0, 0, 1]) #Fizzbuzz
    elif i%5 == 0: return np.array([0, 0, 1, 0]) #Fizz
    elif i%3 == 0: return np.array([0, 1, 0, 0]) #Buzz
    else: return np.array([1, 0, 0, 0]) #Output as is

def init_parameters(num_units, seed=True):
    """
    Initializes parameters for a neural network with given number of layers and shape for each layer
    :param num_hidden_units: Array of number of hidden units for each layer. num_units[0] corresponds to initial input size.
    :return:
    """
    if seed:
        tf.set_random_seed(1)
    parameters = {}
    for i in range(1, len(num_units)):
        parameters["W"+str(i)] = tf.get_variable("W"+str(i), [num_units[i], num_units[i-1]], initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters["b"+str(i)] = tf.get_variable("b"+str(i), [num_units[i], 1], initializer=tf.zeros_initializer())
    return parameters

def training_data():
    """
    Creates training data for NN model.
    :return:
    """
    X = np.array([binary_encode(i) for i in range(2**12)]).T
    Y = np.array([fizzbuzz_encode(i) for i in range(2**12)]).T
    return X,Y


def forward_prop(X, parameters):
    """
    Forward propagation for a 3-layer neural network. RELU->RELU->SOFTMAX
    :param X: Input dataset placeholder of shape input_size x num_examples
    :param parameters: Parameters W1, b1, W2, b2, W3, b3
    :return: Computation graph for output of last linear unit.
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    return Z3

def compute_cost(ZL, Y):
    """
    Compute cost function for model.
    :param ZL: Linear output of last layer in model.
    :param Y: Placeholder variable for "true" labels.
    :return: Tensor for cost computation.
    """
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))



def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs=1500, minibatch_size=32, print_cost=True, seed=True, layers=None):
    """
    Runs a neural network based on provided parameters.
    :param X_train: Training data.
    :param Y_train: Labels for training data.
    :param X_test: Testing data.
    :param Y_test: Labels for testing data.
    :param learning_rate: Learning rate for optimizer.
    :param num_epochs: Number of training iterations to perform.
    :param minibatch_size: Size of minibatches to use.
    :param print_cost: Print current cost after every 100 epochs?
    :param seed: Seed number generator?
    :param layers: List of number of units per hidden layer (e.g. for a 3 layer network with 8 units in first layer, 4 in second, and 2 in third: [8, 4, 2]). Default is [100, 50, output_layer_size].
    :return: (trained parameters, layers, training set accuracy, testing set accuracy)
    """
    if(seed):
        tf.set_random_seed(1)
    (input_size, num_examples) = X_train.shape
    costs = []
    X_place = tf.placeholder(tf.float32, [input_size, None], name="X")
    Y_place = tf.placeholder(tf.float32, [Y_train.shape[0], None], name="Y")
    if layers is None:
        layer_list = [input_size, 100, 50, Y_train.shape[0]]
    else:
        layer_list = [input_size] + layers
    params = init_parameters(layer_list, seed=True)
    Z3 = forward_prop(X_place, params)
    cost_func = compute_cost(Z3, Y_place)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_func)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(num_examples/minibatch_size)
            X_minibatches = np.array_split(X_train, num_minibatches, axis=1)
            Y_minibatches = np.array_split(Y_train, num_minibatches, axis=1)
            for i in range(num_minibatches):
                minibatchX = X_minibatches[i]
                minibatchY = Y_minibatches[i]
                _, minibatch_cost = sess.run([optimizer, cost_func], feed_dict={X_place:minibatchX, Y_place:minibatchY})
                epoch_cost += minibatch_cost / int(num_examples/minibatch_size)
            if print_cost and epoch%100 == 0:
                print("Cost after epoch %i: %f" %(epoch, epoch_cost))
            if print_cost and epoch % 2 == 0:
                costs.append(epoch_cost)
        trained_params = sess.run(params)
        #Plot cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations per 10s')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y_place))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X_place:X_train, Y_place: Y_train}))
        print("Test Accuracy:", accuracy.eval({X_place:X_test, Y_place: Y_test}))
    return

def shuffle_in_unison(a, b):
    """
    Shuffles 2 arrays in unison. i.e. Both before and after shuffling a[i] and b[i] correspond to each other.
    Sourced from https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison.
    :param a:
    :param b:
    :return:
    """
    rng_state = np.random.get_state()
    np.random.shuffle(a.T)
    np.random.set_state(rng_state)
    np.random.shuffle(b.T)


def main():
    (X_train, Y_train) = training_data()
    print("X shape:" + str(X_train.shape))
    print("Y shape:" + str(Y_train.shape))
    breakpoint = int(0.9 * X_train.shape[1])
    shuffle_in_unison(X_train, Y_train)
    X_train, X_test = np.hsplit(X_train, [breakpoint])
    Y_train, Y_test = np.hsplit(Y_train, [breakpoint])
    print("X train shape:" + str(X_train.shape))
    print("Y train shape:" + str(Y_train.shape))
    print("X test shape:" + str(X_test.shape))
    print("Y test shape:" + str(Y_test.shape))
    model(X_train, Y_train, X_test, Y_test)
    return


if __name__ == "__main__":
    main()