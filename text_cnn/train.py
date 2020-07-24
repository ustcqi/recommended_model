#coding:utf8
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys 
import logging
import time

import numpy as np
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from text_cnn import TextCNN
from flags import FLAGS
from data_iterator import DataIterator

cur_time = time.strftime('%Y%m%d-%H%M',time.localtime(time.time()))

log_format = '%(asctime)s - %(levelname)s - %(message)s'
log_file = './log/' + cur_time + '.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format=log_format)

def eval(sess, model):
  iterator = DataIterator(params["test_file"], params["test_data_size"], params["test_data_size"])
  for x, y in iterator:
    accuracy, loss = sess.run([model.accuracy, model.loss], feed_dict={model.input : x, model.label : y})
    logging.info("accuracy:%.6f, test_loss:%.6f" % (accuracy, loss))

def main(params):
  text_cnn = TextCNN(params)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.5
  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(params["epoch"]):
      train_iterator = DataIterator(params["train_file"], params["train_data_size"], params["batch_size"])
      sum_loss = 0.0
      for x, y in train_iterator:
        _, loss, score, label, global_step = sess.run([text_cnn.train_op, \
                                                       text_cnn.loss, \
                                                       text_cnn.score, \
                                                       text_cnn.max_score_label, \
                                                       text_cnn.global_step], \
                                                       feed_dict={text_cnn.input : x, text_cnn.label : y})
        sum_loss += loss
        if global_step % 100 == 0:
          logging.info("global_step:%d, loss:%.6f" % (global_step, sum_loss/100))
          eval(sess, text_cnn)
          sum_loss = 0.0

if __name__ == "__main__":
  params = {"train_file" : "./data/train.txt", 
            "train_data_size" : 300000,
            "test_file" : "./data/test.txt", 
            "test_data_size" : 82688,
            "epoch" : 20,
            "batch_size" : 64}
  main(params)
  sys.exit(0)
