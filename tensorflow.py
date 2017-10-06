import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets("/tmp/data/",one_hot=True)

time_steps = 28
num_units = 128
n_input = 28
lr = 0.001
n_classes = 10
batch_sizes = 128

out_weights  =tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias  = tf.Variable(tf.random_normal([n_classes]))

x=tf.placeholder("float",[None,time_steps,n_input])
y=tf.placeholder("float",[None,n_classes])

input = tf.unstack(x,time_steps,1)

lstm_layer = rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")

prediction = tf.matmul(outputs[-1],out_weights)+out_bias


loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
correct_prediction = tf.equal(tf.arg_max(prediction,1),tf.arg_max(y,1))
accuracy =tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter = 1
    while iter<800:
        batch_x,batch_y = mnist.train.next_batch(batch_sizes)
        batch_x=batch_x.reshape((batch_sizes,time_steps,n_input))
        sess.run(opt,feed_dict={x:batch_x,y:batch_y})
        if iter%10==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",loss)
            print("_________________")
        iter+=1

# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# x = tf.placeholder(tf.float32, [None, 784])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# y = tf.nn.softmax(tf.matmul(x, W) + b)
# y_ = tf.placeholder(tf.float32, [None, 10])
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()
# for _ in range(1000):
#   batch_xs, batch_ys = mnist.train.next_batch(100)
#   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#
