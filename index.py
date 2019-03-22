import numpy as np

size = (2,3)

def to_bipolar(booleanArray):
	result = booleanArray * 1
	result[result < 1] = -1

	return result

def xnor(a, b):
	return np.logical_not(np.bitwise_xor(a, b))

def xnor_matmul(a, b):
	result = []

	for i in range(a.shape[0]):
		aux = []

		for j in range(b.shape[1]):
			k = xnor(a[i, :], b[:, j])
			dot = np.count_nonzero(k) - k[k < 1].shape[0]

			aux.append(dot)

		result.append(aux)

	return np.array(result)

x1 = np.random.choice(a=[False, True], size=size)

def Dense(input_size, num_units):
	params = {
		'weights': np.random.choice(a=[False, True], size=(input_size, num_units)),
		'bias': np.random.choice(a=[False, True], size=(num_units,))
	}

	def forward(x): xnor(x, params['weights'])

	params['forward'] = forward

	return params

l1 = Dense(3, 10)

w = l1['weights']

x1_bp = to_bipolar(x1)
w_bp = to_bipolar(w)

print(x1_bp)
print('')
print(w_bp)
print('')
print(np.matmul(x1_bp, w_bp))
print('')
print(xnor_matmul(x1, w))
