import numpy as np

# hyperparameters
input_size = 2 # dim of input vector
hidden_size = 20 # size of hidden layer of neurons
batch_size = 100
iter_number = 5000
learning_rate = 1e-1

# model parameters
W1 = np.random.randn(hidden_size, input_size)*0.01 # input to hidden
W2 = np.random.randn(2, hidden_size)*0.01 # hidden to output
b1 = np.zeros((hidden_size, 1)) # hidden bias
b2 = np.zeros((2, 1)) # output bias

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x))

def lossFun(inputs, targets):
	"""
	inputs is a list of column vectors
	targets is a list of numbers 0 or 1
	returns the loss and gradients on model parameters
	"""
	x, z1, h, z2, y = {}, {}, {}, {}, {}
	loss = 0
	acc = 0
	# forward pass
	# x --W1,b1--> z1 --sigmoid--> h --W2,b2--> z2 --softmax--> y
	for t in range(len(inputs)):
		x[t] = inputs[t]
		z1[t] = np.dot(W1, x[t]) + b1
		h[t] = sigmoid(z1[t]) # hidden state
		z2[t] = np.dot(W2, h[t]) + b2
		y[t] = softmax(z2[t]) # probabilities of 2 cases
		loss += -np.log(y[t][targets[t],0]) # cross-entropy loss
		acc += 1 if np.argmax(y[t])==targets[t] else 0
	acc /= len(inputs) # accuracy of classification
	# backward pass: compute gradients going backwards
	# x <--W1,b1-- z1 <--sigmoid-- h <--W2,b2-- z2 <--softmax-- y
	dW1, dW2 = np.zeros_like(W1), np.zeros_like(W2)
	db1, db2 = np.zeros_like(b1), np.zeros_like(b2)
	# dhnext = np.zeros_like(hs[0])
	for t in reversed(range(len(inputs))):
		dz2 = np.copy(y[t])
		dz2[targets[t]] -= 1 # backprop into z2 through loss and softmax
		dW2 += np.dot(dz2, h[t].T)
		db2 += dz2 # backprop into W2 and b2
		dh = np.dot(W2.T, dz2) # backprop into h
		dz1 = h[t]*(1-h[t])*dh # backprop into z2 through sigmoid
		dW1 += np.dot(dz1, x[t].T)
		db1 += dz1
	for dparam in [dW1, dW2, db1, db2]:
		np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
	return loss, acc, dW1, dW2, db1, db2


mW1, mW2 = np.zeros_like(W1), np.zeros_like(W2)
mb1, mb2 = np.zeros_like(b1), np.zeros_like(b2) # memory variables for Adagrad
for i in range(iter_number):
	# prepare inputs
	inputs, targets = [], []
	for t in range(batch_size):
		x = np.random.rand(input_size,1) * 4 - 2 # random numbers between -2 and 2 
		y = 1 if np.linalg.norm(x)<1 else 0
		inputs.append(x)
		targets.append(y)
	
	# forward the batch through the net and fetch gradient
	loss, acc, dW1, dW2, db1, db2 = lossFun(inputs, targets)
	if i % 50 == 0: print('iter {}, loss: {}, accuracy: {}'.format(i, loss, acc)) # print progress
	
	# perform parameter update with Adagrad
	for param, dparam, mem in zip([W1, W2, b1, b2], 
                                [dW1, dW2, db1, db2], 
                                [mW1, mW2, mb1, mb2]):
		mem += dparam * dparam
		param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update