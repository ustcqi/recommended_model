##################################################################################
# Author: QiChao
# Update: 2019-10-10
# Description: ftrl online learning algorithm for large-scale sparse features
# Paper: Ad Click Prediction: a view from the tenches
##################################################################################
# coding:utf-8
import sys
import math

import numpy as np
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, recall_score, confusion_matrix

TOLERANCE = 0.000001

class FtrlParams():
  def __init__(self, alpha=0.05, beta=1.0, l1=0.0, l2=0.0):
    self.alpha = alpha
    self.beta = beta
    self.l1 = l1
    self.l2 = l2

class FtrlModel():
  def __init__(self):
    self.n_intercept = 0.0
    self.z_intercept = 0.0
    self.w_intercept = 0.0
    # initial value to each feature is 0
    self.n = {} # fea_sign : n
    self.z = {} # fea_sign : z
    self.w = {} # fea_sign : w
    # num_features = 0
    # FtrlParams params;

# classification
class FtrlProximal:
  
  def __init__(self, ftrl_model, params):
    self._params = params
    self._model = ftrl_model

  def sign(self, x):
    if x < 0:
      return -1.0
    return 1.0 

  def calculate_sigma(self, n, grad, alpha):
    return (math.sqrt(n + grad * grad) - math.sqrt(n)) / alpha

  def calculate_w(self, z, n, alpha, beta, l1, l2):
    s = self.sign(z)
    if (s * z <= l1):
      return 0.0 
    w = (s * l1 - z) / ((beta + math.sqrt(n)) / alpha + l2)
    return w

  def sigmoid(self, x):
    if x <= -35.0:
      return 0.000000000001
    elif x >= 35.0:
      return 0.999999999999
    return 1.0 / (1.0 + math.exp(-x))

  def predict_reg(self, x):
    self._model.w_intercept = self.calculate_w(self._model.z_intercept, self._model.n_intercept, self._params.alpha,
                                          self._params.beta, self._params.l1, self._params.l2)
    wtx = self._model.w_intercept
    # print("w_intercept:", wtx, "z_intercept:", self._model.z_intercept, "n_intercept:", self._model.n_intercept)
    for f in x:
      # init model parameters
      if f not in self._model.w.keys():
        self._model.w[f] = 0.0
        self._model.z[f] = 0.0
        self._model.n[f] = 0.0
      self._model.w[f] = self.calculate_w(self._model.z[f], self._model.n[f], self._params.alpha,
                                     self._params.beta, self._params.l1, self._params.l2)
      wtx += self._model.w[f]
      # print("w[%s]=%.6f" % (f, self._model.w[f]))
      # print(f, self._model.w_intercept, wtx, self._model.w[f])
    return wtx

  def log_loss(self, y, pred):
    if int(y) == 1:
      return -math.log(max(pred, TOLERANCE))
    return -math.log(max(1 - pred, 1 - TOLERANCE))

  def predict(self, x):
    wtx = self.predict_reg(x)
    prob = self.sigmoid(wtx)
    pred_y = 0
    if prob >= 0.5:
      pred_y = 1
    return prob, pred_y
    
  def fit(self, x, y):
    wtx = self.predict_reg(x)
    pred = self.sigmoid(wtx)
    # print(pred, y)
    grad = pred - y
    # print("wtx_reg:", wtx, "grad:", grad, "pred:", pred, "y:", y)
    sigma_intercept = self.calculate_sigma(self._model.n_intercept, grad, self._params.alpha)
    self._model.z_intercept += grad - sigma_intercept * self._model.w_intercept
    self._model.n_intercept += grad * grad

    for f in x:
      sigma = self.calculate_sigma(self._model.n[f], grad, self._params.alpha)
      self._model.z[f] += grad - sigma * self._model.w[f]
      self._model.n[f] += grad * grad

    return self.log_loss(y, pred)

  def weights(self):
    b = self.calculate_w(self._model.z_intercept, self._model.n_intercept, self._params.alpha,
                         self._params.beta, self._params.l1, self._params.l2)
    w = {}
    for f in self._model.w.keys():
      w[f] = self.calculate_w(self._model.z[f], self._model.n[f], self._params.alpha,
                              self._params.beta, self._params.l1, self._params.l2)
    return w, b

  def eval(self, eval_data):
    eval_y = []
    pred_y = []
    for (x, y) in eval_data:
      eval_y.append(y)
      prob, pred = self.predict(x)
      pred_y.append(pred)
    eval_y = np.array(eval_y)
    pred_y = np.array(pred_y)
    auc = roc_auc_score(eval_y, pred_y)
    return auc
  
  def print_params(self):
    print("n_intercept:%.6f" % self._model.n_intercept)
    print("z_intercept:%.6f" % self._model.z_intercept)
    print("w_intercept:%.6f" % self._model.w_intercept)
    # initial value to each feature is 0
    # self.n = {} # fea_sign : n
    #self.z = {} # fea_sign : z
    #self.w = {} # fea_sign : w
