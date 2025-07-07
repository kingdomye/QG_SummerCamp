import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

tf.compat.v1.disable_eager_execution()

N, D_in, H, D_out = 64, 1000, 100, 10
x = tf.compat.v1.placeholder(tf.float32, [None, D_in])
y = tf.compat.v1.placeholder(tf.float32, [None, D_out])

w1 = tf.Variable(tf.compat.v1.random_normal((D_in, H)))
w2 = tf.Variable(tf.compat.v1.random_normal((H, D_out)))

h = tf.matmul(x, w1)
h_relu = tf.maximum(h, tf.zeros(1))
y_pred = tf.matmul(h_relu, w2)

loss = tf.reduce_sum((y - y_pred) ** 2.0)
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

learning_rate = 1e-6
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

x_list, y_list = [], []
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    x_value = np.random.randn(N, D_in)
    y_value = np.random.randn(N, D_out)

    for t in range(500):
        loss_value, _, _ = sess.run([loss, new_w1, new_w2], feed_dict={x: x_value, y: y_value})
        print(loss_value)
        x_list.append(t+1)
        y_list.append(loss_value)

plt.plot(x_list, y_list)
plt.show()

