# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0)

# neural network with 1 layer of 10 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with 784 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 100 lines and 10 columns

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

batch_size = 128
eval_every = 5

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# weights W[784, 10]   784=28*28
W = tf.Variable(tf.zeros([784, 10]))
# biases b[10]
b = tf.Variable(tf.zeros([10]))

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
XX = tf.reshape(X, [-1, 784])

# The model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000# normalized for batches of 100 images,
                                                          # *10 because  "mean" included an unwanted division by 10

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(cross_entropy)

# matplotlib visualisation
allweights = tf.reshape(W, [-1])
allbiases = tf.reshape(b, [-1])
# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# Start training loop
train_loss = []
train_acc = []
test_acc = []

for i in range(500):
    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(batch_size)
    train_dict = {X: batch_X, Y_: batch_Y}

    sess.run(train_step, feed_dict=train_dict)
    temp_train_loss, temp_train_preds = sess.run([cross_entropy , correct_prediction], feed_dict=train_dict)
    temp_train_acc = sess.run(accuracy, feed_dict=train_dict)
    acc_and_loss = [(i+1), temp_train_loss, temp_train_acc]
    acc_and_loss = [np.round(x,2) for x in acc_and_loss]
    print('Generation # {}. Train Loss: {:.2f}. Train Acc : {:.2f}'.format(*acc_and_loss))
    if((i) % 10 == 0):
        train_loss.append(temp_train_loss)
        train_acc.append(temp_train_acc * 100)
    
    
a = int(len(mnist.test.images))
for i in range(int(a/20)):
    eval_X, eval_Y = mnist.test.next_batch(20)
    test_dict = {X: eval_X, Y_: eval_Y}
    
    test_preds = sess.run(correct_prediction, feed_dict=test_dict)
    temp_test_acc= sess.run(accuracy, feed_dict=test_dict)
  #  print(i, temp_test_acc)
    # Record and print results
    if((i) % 10 == 0):
        test_acc.append(temp_test_acc * 100)

# Matlotlib code to plot the loss and accuracies
eval_indices = range(0, 500, 10)
# Plot loss over time
plt.plot(eval_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()       

# Plot train and test accuracy
plt.ylim(0, 100)
plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(eval_indices, test_acc, 'r-', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(2000+1): training_step(i, i % 50 == 0, i % 10 == 0)


# final max test accuracy = 0.9268 (10K iterations). Accuracy should peak above 0.92 in the first 2000 iterations.
