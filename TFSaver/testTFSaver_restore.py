import tensorflow as tf

tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  #模型的恢复用的是restore()函数，它需要两个参数restore(sess, save_path)，save_path指的是保存的模型路径。我们可以使用tf.train.latest_checkpoint()
  #来自动获取最后一次保存的模型。
  #saver.restore(sess, tf.train.latest_checkpoint('/tmp'))
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())

