#coding:utf8
import tensorflow as tf

class Word2Vec(object):

  def __init__(self):
    self.build_model()

  def build_model(self):
    self.inputs = tf.placeholder(tf.int32, shape=[None, 1])
    self.labels = tf.placeholder(tf.int32, shape=[None, 1])

    global_step = tf.train.get_or_create_global_step()

    with tf.device('/cpu:0'):
      self.embedding = tf.Variable(tf.random_uniform([FLAGS.vocab_size, FLAGS.embedding_dim]), name="embedding")
    
    self.input_embeddings = tf.nn.embedding_lookup(self.embedding, self.inputs)
    nce_weights = tf.Variable(tf.truncated_normal([FLAGS.vocab_size, FLAGS.embedding_dim], stddev=1.0 / math.sqrt(FLAGS.embedding_dim)))
    nce_bias = tf.Variable(tf.zeros([FLAGS.vocab_size]), dtype=tf.float32)

    with tf.name_scope("loss"):
      loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_bias, inputs=self.input_embedding, num_sampled=FLAGS.num_sampled, num_classes=FLAGS.vocab_size))
    
    with tf.name_scope("train"):
      optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
      train_op = optimizer.minimize(loss, global_step=global_step)
