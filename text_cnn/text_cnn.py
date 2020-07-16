#coding:utf8
import tensorflow as tf
from flags import FLAGS

class TextCNN(object):

  def __init__(self, params):
    self.params = params
    self.build_model(params)

  def build_model(self, params):
    self.input = tf.placeholder(tf.int32, [None, FLAGS.max_input_length], name="input")
    self.label = tf.placeholder(tf.float32, [None, FLAGS.output_num], name="label")
    self.global_step = tf.Variable(0, trainable=False)

    with tf.device("/cpu:0"), tf.name_scope("embedding"):
      self.embedding = tf.Variable(tf.random_uniform([FLAGS.vocab_size, FLAGS.embedding_dim]))
      self.input_embedding = tf.nn.embedding_lookup(self.embedding, self.input, name="input_embedding")
      self.input_embedding_expanded = tf.expand_dims(self.input_embedding, -1)

    pooled_outputs = []
    for i, filter_size in enumerate((FLAGS.filters_size).split(',')):
      filter_size = int(filter_size)
      with tf.name_scope("convolution_%d" % filter_size):
        filters = tf.Variable(tf.truncated_normal(shape=[filter_size, FLAGS.embedding_dim, 1, FLAGS.filter_num]), name="conv_filter")
        conv = tf.nn.conv2d(self.input_embedding_expanded, filters, strides=[1, 1, 1, 1], padding="VALID", name="conv") 
        b = tf.Variable(tf.constant(0.0, shape=[FLAGS.filter_num]))
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        pooled = tf.nn.max_pool(h, ksize=[1, FLAGS.max_input_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding="VALID")
        # print(pooled.get_shape().as_list())
        pooled_outputs.append(pooled)
    
    filter_num_total = FLAGS.filter_num * len((FLAGS.filters_size).split(','))
    self.h_pooled = tf.concat(pooled_outputs, axis=3)
    self.h_pooled = tf.reshape(self.h_pooled, [-1, filter_num_total])
    
    with tf.name_scope("output"):
      self.W = tf.Variable(tf.truncated_normal([filter_num_total, FLAGS.output_num]), name="last_weight")
      self.b = tf.Variable(tf.constant(0.1, shape=[FLAGS.output_num]), name="bias")
      self.output = tf.add(tf.matmul(self.h_pooled, self.W), self.b, name="output")

    with tf.name_scope("regularizer"):
      self.l2_loss = FLAGS.l2_lambda * tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b)

    with tf.name_scope("classfication"):
      if FLAGS.output_num == 1:
        self.score = tf.nn.sigmoid(self.output, name="bi_score")
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.output, name="bi_loss"))
      else:
        self.score = tf.nn.softmax(self.output, name="multi_score") 
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.output, name="multi_loss"))
        self.max_score_label = tf.argmax(self.score, axis=1, name="max_score_label")
        self.correct_predictions = tf.equal(self.max_score_label, tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, 'float'), name="accuracy")
        self.num_correct = tf.reduce_sum(tf.cast(self.correct_predictions, 'float'), name="num_correct")
    
    with tf.name_scope("train"):
      self.learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
      self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)
