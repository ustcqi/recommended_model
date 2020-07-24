#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import collections
import datetime
import logging
import os.path
import multiprocessing
import gensim
import numpy

grams_file = sys.argv[1]
idea_count = int(sys.argv[2])
all_vector_txt = sys.argv[3]
all_model_txt = sys.argv[4]
iter = int(sys.argv[5])
size = int(sys.argv[6])
window = int(sys.argv[7])
min_count = int(sys.argv[8])

timestamp = timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
all_model_txt = all_model_txt + '.' + timestamp
# all_vector_txt = all_vector_txt

#program = os.path.basename(sys.argv[0])
log_file = './log/doc2vec.log.' + timestamp
logging.basicConfig(filename=log_file, format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
logging.info("running %s" % ' '.join(sys.argv))

start = time.time()
#documents = gensim.models.doc2vec.TaggedLineDocument('/serving/libin/env_model/new/account2vec/' + grams_file)
documents = gensim.models.doc2vec.TaggedLineDocument(grams_file)
end = time.time()
print 'cost time< doc >:', end - start
logging.info("generate documents cost time %.6f" % (end-start))
start = time.time()
model = gensim.models.doc2vec.Doc2Vec(documents, iter=iter, size=size, window=window, min_count=min_count, workers = multiprocessing.cpu_count())
end = time.time()
print 'cost time< model >:', end - start
logging.info("training time cost %.6f" % (end-start))
model.save(all_model_txt)

logging.info("model.docves number=%d" % len(model.docvecs))
#idea_count = 2828901
with open(all_vector_txt, 'w') as fd:
  for num in range(0, idea_count):
    docvec = model.docvecs[num]
    numpy.savetxt(fd, numpy.reshape(docvec, [1, size]), '%.7f')
logging.info("doc2vec already generated.")
#sys.exit()
