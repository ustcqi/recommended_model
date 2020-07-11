#coding:utf8
import tensorflow as tf

class TextCnn(object):

  def __init__(self, params):
    self.params = params

  def build_model(self):
    self.input = tf.placeholder(tf.int32, [None, FLAGS.max_input_length], name="input")
