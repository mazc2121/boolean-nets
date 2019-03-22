import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from ga import GA
from es import ES
import random
from random import randint
from scipy.sparse import rand

np.set_printoptions(suppress=True)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((-1, 784)) / 255
x_test = x_test.reshape((-1, 784)) / 255

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y_train)

y_train = enc.transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

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

def binary_rand(shape):
	return np.random.choice(a=[False, True], size=shape)

def normal_rand(shape):
	return np.random.uniform(-0.5, 0.5, size=shape)

def binary_mutation(value):
	m = rand(value.shape[0], value.shape[1], density=0.1).todense() > 0

	return np.logical_or(value, m)

def normal_mutation(value):
	m = rand(value.shape[0], value.shape[1], density=0.1).todense()

	return value + m * (1 if randint(0, 1) > 0.5 else -1)

def sparse_rand(shape):
	m = rand(shape[0], shape[1], density=0.1).todense()

	m[m > 0] = random.uniform(-0.5, 0.5)

	return m

def XnorDense(input_size, num_units):
	params = {
		'weights': np.empty((input_size, num_units))
	}

	def forward(x, p): 
		return xnor_matmul(x, p['weights'])

	return { 'params': params, 'forward': forward }

def Dense(input_size, num_units, activation='relu'):
	params = {
		'weights': normal_rand((input_size, num_units)),
		'bias': np.zeros((1, num_units))
	}

	def forward(x, p):
		x = np.matmul(x, p['weights']) + p['bias']

		if activation == 'relu':
			return np.maximum(x, 0)
		elif activation == 'softmax':
			return softmax(x, axis=1)

	return { 'params': params, 'forward': forward }

BATCH_SIZE = 32
INPUT_SIZE = 784
OUTPUT_SIZE = 10
NUM_UNITS = 128
POP_SIZE = 200
NUM_PARENTS = 20

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

model.push(XnorDense(INPUT_SIZE, NUM_UNITS))
model.push(XnorDense(NUM_UNITS, OUTPUT_SIZE))

normal_model = Model()

normal_model.push(Dense(INPUT_SIZE, NUM_UNITS))
normal_model.push(Dense(NUM_UNITS, NUM_UNITS))
normal_model.push(Dense(NUM_UNITS, OUTPUT_SIZE, activation='softmax'))

# opt = GA(pop_size=POP_SIZE, num_parents=NUM_PARENTS, \
#  fitness_func=log_loss, rand_func=normal_rand, mutation_func=normal_mutation)

opt = ES(pop_size=POP_SIZE, fitness_func=log_loss, rand_func=sparse_rand)

ini_idx = 0
end_idx = BATCH_SIZE

while ini_idx < y_train.shape[0]:
	batch_xs = x_train[ini_idx:end_idx]
	batch_ys = y_train[ini_idx:end_idx]

	opt.fit(normal_model, batch_xs, batch_ys)

	normal_model.set_params(opt.best)

	pred = normal_model.forward(x_test)

	pred = np.argmax(pred, axis=1)
	target = np.argmax(y_test, axis=1)

	print('fitness', opt.bestFitness, 'accuracy', np.mean(pred == target))

	ini_idx += BATCH_SIZE
	end_idx += BATCH_SIZE