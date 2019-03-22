import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import log_loss
from ga import GA

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

tf.logging.set_verbosity(old_v)

def to_bipolar(booleanArray):
	result = booleanArray * 1
	result[result < 1] = -1

	return result

def xnor(a, b):
	return np.logical_not(np.bitwise_xor(a, b))

def xnor_matmul(a, b):
	result = []
	majority = a.shape[1] * 0.5

	for i in range(a.shape[0]):
		aux = []

		for j in range(b.shape[1]):
			dot = np.count_nonzero(xnor(a[i, :], b[:, j])) > majority

			aux.append(dot)

		result.append(aux)

	return np.array(result)

def Dense(input_size, num_units):
	params = {
		'weights': np.empty((input_size, num_units))
		#'weights': np.random.choice(a=[False, True], size=(input_size, num_units))
	}

	def forward(x, p): 
		return xnor_matmul(x, p['weights'])

	return { 'params': params, 'forward': forward }

BATCH_SIZE = 32
INPUT_SIZE = 784
NUM_UNITS = 500
EPOCHS = 1
POP_SIZE = 100
NUM_MATING = 10

class Model:
	def __init__(self):
		self.layers = []

	def push(self, layer):
		self.layers.append(layer)

	def forward(self, x):
		for l in self.layers:
			x = l['forward'](x, l['params'])

		return x

	def get_params(self):
		return [ l['params'] for l in self.layers ]

	def set_params(self, params):
		for i in range(len(self.layers)):
			self.layers[i]['params'] = params[i]


model = Model()

model.push(Dense(INPUT_SIZE, NUM_UNITS))
model.push(Dense(NUM_UNITS, 10))

opt = GA(pop_size=POP_SIZE, num_mating=NUM_MATING, fitness_func=log_loss)

for epoch in range(EPOCHS):

	while True:
		batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)

		if not batch_xs.any():
			break

		opt.fit(model, batch_xs > 0, batch_ys > 0)
		break