#coding:utf8
import tensorflow as tf

from attention_layer import attention, kv_attention, self_attention

def test_attention(sess):
  query = tf.constant([0, 1, 2, 3, 4, 5], shape=[2, 1, 3], dtype=tf.float32)
  keys = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], shape=[2, 2, 3], dtype=tf.float32)
  keys_length = tf.constant([2, 1])
  print("query:" , sess.run(query))
  print("keys:", sess.run(keys))
  print("keys_length:", sess.run(keys_length))
  output = attention(query, keys, keys_length, sess)
  print(sess.run(output))

def test_kv_attention(sess):
  query = tf.constant([0, 1, 2, 3, 4, 5], shape=[2, 1, 3], dtype=tf.float32)
  keys = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], shape=[2, 2, 3], dtype=tf.float32)
  keys_length = tf.constant([2, 1])
  values = tf.constant([7, 6, 5, 4, 3, 2, 1, 0], shape=[2, 2, 2], dtype=tf.float32)
  print("query:" , sess.run(query))
  print("keys:", sess.run(keys))
  print("keys_length:", sess.run(keys_length))
  print("values:", sess.run(values))
  output = kv_attention(query, keys, keys_length, values, sess)
  print(sess.run(output))

def test_self_attention():
  inputs = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], shape=[2, 4, 3], dtype=tf.float32)
  inputs_length = tf.constant([2, 3])
  print("inputs:" , sess.run(inputs))
  print("inputs_length:" , sess.run(inputs_length))
  output = self_attention(inputs, inputs_length, 5, sess)
  print(sess.run(output))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
test_attention(sess)
# test_kv_attention(sess)
# test_self_attention()
