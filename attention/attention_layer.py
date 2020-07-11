#coding:utf8
import tensorflow as tf


"""
basic attention
mode 1: query [B H] and keys [B T H]
 steps: 
  1.1 keys * query
mode 2: query, keys and values
mode 3: multi-head, queries, keys and values


global-attention and lcoal-attention


blog:
1. https://blog.csdn.net/qq_37394634/article/details/102679096
2. https://www.zhihu.com/question/68482809
3. https://tobiaslee.top/2017/08/15/Attention-Mechanism-%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/
"""
# basic attention pattern
def attention(queries, keys, keys_length, sess=None):
  """
   arguments
    queries : [B T_q H]
    keys : [B T H]
    keys_length : [B]
   return: attention_output [B T_q H]

   steps:
   1. output = Queries * transpose(Keys) -> [B T_q T]
   2. mask output
   3. alhpa = softmax(Keys * Query) / sqrt(H)
   4. alpha matmul keys -> [B T H] 
  """
  [B, T_q, H] = queries.get_shape().as_list()
  [B, T, H] = keys.get_shape().as_list()
  output = tf.matmul(queries, tf.transpose(keys, [0, 2, 1])) / H**0.5# [B T_q T]
  # print("keys * queries: ", sess.run(output))
  mask_output = mask(output, keys_length)

  # attention operation
  weighted_output = tf.nn.softmax(mask_output) # [B T_q T]
  # print("weighted_output: ", sess.run(weighted_output))

  # weighted_output = tf.expand_dims(weighted_output, 1)
  # keys -> mask_keys
  attention_output = tf.matmul(weighted_output, keys) # [B T_q H]
  return attention_output

def kv_attention(queries, keys, keys_length, values, sess=None):
  """
   arguments
    queries : [B T_q H] 特例 T_q=1
    keys : [B T H]
    keys_length : [B]
    values : [B T Hv]
   return: attention_output
  """
  [B, T_q, H] = queries.get_shape().as_list()
  [B, T, H] = keys.get_shape().as_list()
  [B, T, Hv] = values.get_shape().as_list()
  output = tf.matmul(queries, tf.transpose(keys, [0, 2, 1])) / H**0.5 # [B T_q T]
  # print("keys * query: ", sess.run(output))
  mask_output = mask(output, keys_length)
 
  # attention operation
  weighted_output = tf.nn.softmax(mask_output)
  # print("weighted_output: ", sess.run(weighted_output))

  # keys -> mask_keys
  attention_output = tf.matmul(weighted_output, values)
  # print(attention_output.get_shape().as_list())
  return attention_output

"""
self-attention
why use self-attention? 
1) convolution net and recurrent net can't solve long-term dependencies?
convolution is based on local encoding(n-gram), rnn can only solve short-term dependencies because of vanishing gradient.
2) full-connect net can solve long-term dependencies but can't solve variable-length input.
so, we can use attention mechanism to model the weight of connection dynamiclly.
steps:
1. Query = Wq * X
2. Key = Wk * X
3. V = Wv * X
4. same as basic attention
"""

def self_attention(inputs, inputs_length, output_dims, sess=None):
  """
    inputs : [B T H]
    inputs_length : [B] 
    output_dims :  N
  return 
  """
  [B, T, H] = inputs.get_shape().as_list()
  Wq = tf.random_uniform(shape=[H, output_dims], minval=-1.0, maxval=1.0, dtype=tf.float32)
  # Wq = tf.layers.dense(inputs, output_dims, use_bias=True)
  Wq = tf.expand_dims(Wq, 0)
  Wq = tf.tile(Wq, [B, 1, 1])
  queries = tf.matmul(inputs, Wq)

  Wk = tf.random_uniform(shape=[H, output_dims],  minval=-1.0, maxval=1.0, dtype=tf.float32)
  # Wk = tf.layers.dense(inputs, output_dims, use_bias=True)
  Wk = tf.expand_dims(Wk, 0)
  Wk = tf.tile(Wk, [B, 1, 1])
  keys = tf.matmul(inputs, Wk)

  Wv = tf.random_uniform(shape=[H, output_dims], minval=-1.0, maxval=1.0, dtype=tf.float32)
  # Wv = tf.layers.dense(inputs, output_dims, use_bias=True)
  Wv = tf.expand_dims(Wv, 0)
  Wv = tf.tile(Wv, [B, 1, 1])
  values = tf.matmul(inputs, Wv)
  
  outputs = kv_attention(queries, keys, inputs_length, values, sess)
  return outputs

def mask(inputs, inputs_length):
  """
    inputs : [B T_q T]
    inputs_length : [B]
  """
  [B, T_q, T] = inputs.get_shape().as_list()

  # mask inputs before softmax
  keys_mask = tf.sequence_mask(inputs_length, T)
  keys_mask = tf.expand_dims(keys_mask, 1) # [B 1 T]
  keys_mask = tf.tile(keys_mask, [1, T_q, 1]) # [B T_q T]
  paddings = tf.ones_like(inputs) * (-2**32 + 1)
  
  mask_output = tf.cast(tf.where(keys_mask, inputs, paddings), tf.float32) # [B T_q T]
  return mask_output

def scaled_dot_product_attention(Q, K, V, keys_length):
  """
    Q : [B T_q, H]
    K : [B T_k, H]
    V : [B T_k, Hv]
    keys_length : [B]
  return outputs [B T_q Hv]
  """
  with tf.variable_scope("scaled_dot_product_attention", tf.AUTO_REDUSE):
    H = Q.get_shape().as_list()[-1]
    # [B*h T_q T_k]
    outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
    outputs /= H**0.5
    outputs = mask(outputs, keys_length)
    # keys attention distribution
    outputs = tf.nn.softmax(outputs)
    # [B T_q Hv]
    outputs = tf.matmul(outputs, V)
    return outputs

"""
multihead-self-attention, queries = keys = values
queries [B T_q H]
keys [B T_k H]
values [B T_k H]
"""
def multihead_attention(queries, keys, values, keys_length, num_heads):
  """
    queries : [B, T_q, H]
    keys : [B T_k, H]
    values : [B T_k, H]
  """
  [B, T_q, H] = queries.get_shape().as_list()[-1]
  with tf.variable_scope("multihead_attention", tf.AUTO_REUSE):
    Q = tf.layers.dense(queries, H, use_bias=True)
    K = tf.layers.dense(keys, H, use_bias=True)
    V = tf.layers.dense(values, H, use_bias=True)

    # [B*h T_q H/h]
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

    # [B*h T_q Hv] Hv=H/h
    outputs = scaled_dot_product_attention(Q_, K_, V_, keys_length)
    # [B
    outputs = tf.concat(tf.split(outputs, num_head, axis=0), axis=2)
    return output
