import tensorflow as tf  
import tensorflow
import tf_test
import tensorflow.contrib.slim as slim  
tf.reset_default_graph() 
#BATCH_SIZE = 3  


train_log_dir ="log/"
if not tf.gfile.Exists(train_log_dir):
  tf.gfile.MakeDirs(train_log_dir)

with tf.Graph().as_default():
  # Set up the data loading:
  images, labels = tf_test.read_and_decode("train.tfrecords")
  img_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                batch_size=3, capacity=2000,
                                                min_after_dequeue=1000)
  label_batch=tf.reshape(label_batch, [3])

  labels=slim.one_hot_encoding(label_batch,num_classes=2)

  # Define the model:
  predictions,_ = tensorflow.contrib.slim.nets.vgg.vgg_16(img_batch,2, is_training=True)

  # Specify the loss function:
  slim.losses.softmax_cross_entropy(predictions, labels)

  total_loss = slim.losses.get_total_loss()
  tf.summary.scalar('losses/total_loss', total_loss)

  # Specify the optimization scheme:
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

  # create_train_op that ensures that when we evaluate it to get the loss,
  # the update_ops are done and the gradient updates are computed.
  train_tensor = slim.learning.create_train_op(total_loss, optimizer)

  # Actually runs training.
  slim.learning.train(train_tensor, train_log_dir)