import numpy as np

size = (10,)

def to_bipolar(booleanArray):
	result = booleanArray * 1
	result[result == 0] = -1

	return result

def xnor(a, b):
	return np.logical_not(np.bitwise_xor(a, b))

x1 = np.random.choice(a=[False, True], size=size)
x2 = np.random.choice(a=[False, True], size=size)

x1_bp = to_bipolar(x1)
x2_bp = to_bipolar(x2)

print(x1_bp)
print(x2_bp)
print(x1_bp * x2_bp)

print(xnor(x1_bp, x2_bp))