import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("max_input_length", 32, "maximum input text length")
flags.DEFINE_integer("vocab_size", 500000, "embedding table vocab size")
flags.DEFINE_integer("embedding_dim", 50, "embedding table latent vector dimenson")
flags.DEFINE_string("filters_size", "2,3,5", "convolution layer filters size")
flags.DEFINE_integer("filter_num", 32, "filter number")
flags.DEFINE_integer("output_num", 15, "binary classfication:1, multi_classification:class_num")
flags.DEFINE_float("l2_lambda", 0.1, "l2 regularization coefficient")

flags.DEFINE_float("learning_rate", 0.001, "")
