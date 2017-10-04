import numpy as np
import tensorflow as tf

# hyperparameters
input_size = 2 # dim of input vector
hidden_size = 20 # size of hidden layer of neurons
batch_size = 100
iter_number = 5000
learning_rate = 1e-1

# inputs and targets
inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.float32, [None, 2])

# model parameters
W1 = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=0.01)) # input to hidden
W2 = tf.Variable(tf.random_normal([hidden_size, 2], stddev=0.01)) # hidden to output
b1 = tf.Variable(tf.zeros([hidden_size])) # hidden bias
b2 = tf.Variable(tf.zeros([2])) # output bias

# hidden variables
z1 = tf.matmul(inputs, W1) + b1
h = tf.nn.sigmoid(z1)
z2 = tf.matmul(h, W2) + b2
y = tf.nn.softmax(z2)

# loss and optimizer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(targets * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=1e-8).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(iter_number):
	# prepare inputs
	inputs_batch = (np.random.rand(batch_size, input_size)*4-2).astype('float32')
	targets_batch = np.zeros([batch_size,2]).astype('float32')
	mask = (np.linalg.norm(inputs_batch, axis=1)<1).astype(int)
	for t in range(batch_size):
		targets_batch[t][mask[t]] = 1 # one-hot encoding
	# run training step
	_, loss, acc = sess.run([train_step, cross_entropy, accuracy], feed_dict={inputs: inputs_batch, targets: targets_batch})
	# print progress
	if i % 50 == 0: print('iter {}, loss: {:.4f}, accuracy: {:.2f}'.format(i, loss, acc)) 