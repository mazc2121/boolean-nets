import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Dense

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((-1, 784)) / 255
x_test = x_test.reshape((-1, 784)) / 255

y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y_train)

y_train = enc.transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()

BATCH_SIZE = 32
INPUT_SIZE = 784
OUTPUT_SIZE = 10
NUM_UNITS = 128

X = tf.placeholder("float", [None, INPUT_SIZE])
Y = tf.placeholder("float", [None, OUTPUT_SIZE])

l1 = Dense(NUM_UNITS, activation='relu')
l2 = Dense(NUM_UNITS, activation='relu')
l3 = Dense(OUTPUT_SIZE, activation='softmax')

model = l1(X)
model = l2(model)
model = l3(model)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
	logits=model, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08) \
	.minimize(loss_op)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	ini_idx = 0
	end_idx = BATCH_SIZE

	while ini_idx < y_train.shape[0]:
		batch_xs = x_train[ini_idx:end_idx]
		batch_ys = y_train[ini_idx:end_idx]

		loss,_ = sess.run([ loss_op, optimizer ], feed_dict={ X: batch_xs, Y: batch_ys })

		pred = sess.run(model, feed_dict={ X: x_test })

		pred = np.argmax(pred, axis=1)
		target = np.argmax(y_test, axis=1)

		print('loss', loss, 'accuracy', np.mean(pred == target))

		ini_idx += BATCH_SIZE
		end_idx += BATCH_SIZE
