#coding:utf8
import numpy as np
from flags import FLAGS

class DataIterator():

  def __init__(self, data_file, data_size, batch_size):
    self.batch_size = batch_size
    self.idx = 0
    self.X, self.Y = self.get_data(data_file, data_size)
    self.batch_num = len(self.Y) // self.batch_size

  def get_data(self, data_file, data_size):
    X = np.zeros((data_size, FLAGS.max_input_length), dtype=int)
    Y = np.zeros((data_size, FLAGS.output_num), dtype=float)
    idx = 0
    with open(data_file) as in_file:
      for line in in_file:
        items = line.strip('\n').split('\t')
        label = int(items[0])
        Y[idx][label] = 1
        features = items[2].split(' ')
        for i in range(len(features)):
          if i < FLAGS.max_input_length:
            X[idx][i] = int(features[i])
        idx += 1
    return X[:idx], Y[:idx]

  def __iter__(self):
    return self

  def __next__(self):
    if self.idx == self.batch_num:
      raise StopIteration
    si = self.idx * self.batch_size
    ei = (self.idx + 1) * self.batch_size
    self.idx += 1
    return self.X[si:ei], self.Y[si:ei]

if __name__ == "__main__":
  for x, y in DataIterator("./data/test.txt", 82688, 4):
    print(x, y)
