import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - (tanh(x))**2


class NeuralNetwork:

    # Initialize the neural network 
    def __init__(self, layers): 

	# Network Structure (without bias neurons) layers = [2,2,1] 
        # The usual XOR neural network 
	# len(layers) = 3 (3 layers, input/hidden/output
	#	print 'the number of layers %5d' % len(layers)


	# Activation Function Selection (tanh is the choice, but choose others)
        self.activation = tanh
        self.activation_prime = tanh_prime

        # Set weights
	# range of weight values (-1,1): 2*random-1 since random gives (0,1)

        self.weights = []


        # First: weight between input and hidden layers - random((2+1, 2+1)) : 3 x 3
	for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
	    # *** Silly for-loop, since i can take value of 1 only due to range(1,2)

        # Second: weight between hidden and output layer - random((2+1, 1)) : 3 x 1
            r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
            self.weights.append(r)

	#print "Initial Random Weights"
        #print self.weights

    # Traning the neural network
    # X (input features - vector ) and y (output - scalar) are the training sets 
    #  
    def fit(self, X, y, epochs, learning_rate=0.6): 

	#print "Input data (X, y)"
        #print X
        #print y

        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)

        #print "Input data (X, y) after adding bias neurons"
        #print X
        #print y


	# Now Training starts
        # How many times that we pass through all four data points = epochs
 
        for k in range(epochs):
            if k % 10 == 0: print 'epochs:', k
            
            i = np.random.randint(X.shape[0])
            a = [X[i]]

	    #print "which data point now?"
            #print i

            #print "Tracking the neuron output"
            #print a


	    # *********************************************
	    #  Forward Computation
	    # *********************************************
	    #print "How many layers %d" % len(self.weights)
	      
            for l in range(len(self.weights)):

		    #print "What is l=%d" % l
		    #print "What is neuron output"
		    #print a[l]
		    #print "Weight"
		    #print self.weights[l]

                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)

		    #print "Tracking the neuron output"
	            #print a


	    # **********************************************
	    # Backward Computation
            # **********************************************

            # output layer result - error 
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer) 
            # len(a)=3 

            for l in range(len(a) - 2, 0, -1): 
		print "l in backward portion %d" %len(a)

                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)



    def predict(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == '__main__':

    nn = NeuralNetwork([2,2,1])

    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([0, -0.46, 0.8, -0.55])

    nn.fit(X, y, 10000)

    for e in X:
        print(e,nn.predict(e))
