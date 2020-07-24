# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""For training NMT models."""
from __future__ import print_function

import collections
import math
import os
import random
import time

import tensorflow as tf

from . import attention_model
from . import gnmt_model
from . import inference
from . import model as nmt_model
from . import model_helper
from .utils import iterator_utils
from .utils import misc_utils as utils
from .utils import nmt_utils
from .utils import vocab_utils

from tensorflow.python import debug as tfdbg

utils.check_tensorflow_version()

__all__ = [
    "create_train_model", "create_eval_model", "run_sample_decode",
    "run_internal_eval", "run_external_eval", "run_full_eval", "train",
]


class TrainModel(
    collections.namedtuple("TrainModel", ("graph", "model", "iterator",
                                          "skip_count_placeholder"))):
  pass


def create_train_model(
    model_creator, hparams, scope=None, single_cell_fn=None,
    model_device_fn=None):
  """Create train graph, model, and iterator."""
  src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
  # tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
  # NOTE(qichao)
  tgt_file = src_file
  src_vocab_file = hparams.src_vocab_file
  # tgt_vocab_file = hparams.tgt_vocab_file
  # NOTE(qichao)
  tgt_vocab_file = src_vocab_file

  graph = tf.Graph()

  with graph.as_default():
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)

    src_dataset = tf.contrib.data.TextLineDataset(src_file)
    # tgt_dataset = tf.contrib.data.TextLineDataset(tgt_file)
    # NOTE (qichao)
    tgt_dataset = src_dataset
    skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

    iterator = iterator_utils.get_clk_iterator(
        src_dataset,
        tgt_dataset,
        src_vocab_table,
        tgt_vocab_table,
        batch_size=hparams.batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        src_reverse=hparams.src_reverse,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=hparams.src_max_len,
        tgt_max_len=hparams.tgt_max_len,
        num_threads=1,
        skip_count=skip_count_placeholder)

    # Note: One can set model_device_fn to
    # `tf.train.replica_device_setter(ps_tasks)` for distributed training.
    with tf.device(model_device_fn):
      model = model_creator(
          hparams,
          iterator=iterator,
          mode=tf.contrib.learn.ModeKeys.TRAIN,
          source_vocab_table=src_vocab_table,
          target_vocab_table=tgt_vocab_table,
          scope=scope,
          single_cell_fn=single_cell_fn)

  return TrainModel(
      graph=graph,
      model=model,
      iterator=iterator,
      skip_count_placeholder=skip_count_placeholder)


class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model", "src_file_placeholder",
                            "tgt_file_placeholder", "iterator"))):
  pass


def create_eval_model(model_creator, hparams, scope=None, single_cell_fn=None):
  """Create train graph, model, src/tgt file holders, and iterator."""
  src_vocab_file = hparams.src_vocab_file
  # tgt_vocab_file = hparams.tgt_vocab_file
  tgt_vocab_file = hparams.src_vocab_file
  graph = tf.Graph()

  with graph.as_default():
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)
    src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
    # tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
    # NOTE(qichao)
    tgt_file_placeholder = src_file_placeholder
    src_dataset = tf.contrib.data.TextLineDataset(src_file_placeholder)
    # tgt_dataset = tf.contrib.data.TextLineDataset(tgt_file_placeholder)
    # NOTE(qichao)
    tgt_dataset = src_dataset
    iterator = iterator_utils.get_clk_iterator(
        src_dataset,
        tgt_dataset,
        src_vocab_table,
        tgt_vocab_table,
        hparams.batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        src_reverse=hparams.src_reverse,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=hparams.src_max_len_infer,
        tgt_max_len=hparams.tgt_max_len_infer)
    model = model_creator(
        hparams,
        iterator=iterator,
        mode=tf.contrib.learn.ModeKeys.EVAL,
        source_vocab_table=src_vocab_table,
        target_vocab_table=tgt_vocab_table,
        scope=scope,
        single_cell_fn=single_cell_fn)
  return EvalModel(
      graph=graph,
      model=model,
      src_file_placeholder=src_file_placeholder,
      tgt_file_placeholder=tgt_file_placeholder,
      iterator=iterator)

def run_eval_clk(
    eval_model, eval_sess, model_dir, hparams, summary_writer):
  """Compute internal evaluation (perplexity) for both dev / test."""
  with eval_model.graph.as_default():
    loaded_eval_model, global_step = model_helper.create_or_load_model(
        eval_model.model, model_dir, eval_sess, "eval")
  if hparams.test_prefix:
    test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
    test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
    test_eval_iterator_feed_dict = {
        eval_model.src_file_placeholder: test_src_file,
        eval_model.tgt_file_placeholder: test_tgt_file
    }
    _internal_eval_clk(loaded_eval_model, global_step, eval_sess,
                       eval_model.iterator, test_eval_iterator_feed_dict,
                       summary_writer, "test", hparams.hidden_state_file)

def _internal_eval(model, eval_step, sess, iterator, iterator_feed_dict,
                   summary_writer, label):
  # Computing perplexity
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
  # NOTE(qichao)
  ppl = model_helper.compute_perplexity(model, sess, label)
  utils.add_summary(summary_writer, eval_step, "%s_ppl", ppl)

def run_internal_eval(
    eval_model, eval_sess, model_dir, hparams, summary_writer):
    with eval_model.graph.as_default():
      loaded_eval_model, eval_step = model_helper.create_or_load_model(
          eval_model.model, model_dir, eval_sess, "eval")
    utils.print_out("eval_step %d compute_perplexity" % eval_step)
    if hparams.test_prefix:
      test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
      test_eval_iterator_feed_dict = {
          eval_model.src_file_placeholder: test_src_file,
	  eval_model.tgt_file_placeholder: test_src_file
      }
      _internal_eval(loaded_eval_model, eval_step, eval_sess,
                     eval_model.iterator, test_eval_iterator_feed_dict,
		     summary_writer, "test")

def train(hparams, scope=None, target_session="", single_cell_fn=None):
  """Train a translation model."""
  log_device_placement = hparams.log_device_placement
  out_dir = hparams.out_dir
  num_train_steps = hparams.num_train_steps
  steps_per_stats = hparams.steps_per_stats
  steps_per_external_eval = hparams.steps_per_external_eval
  steps_per_eval = 10 * steps_per_stats
  if not steps_per_external_eval:
    steps_per_external_eval = 5 * steps_per_eval

  if not hparams.attention:
    model_creator = nmt_model.Model
  elif hparams.attention_architecture == "standard":
    model_creator = attention_model.AttentionModel
  elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
    model_creator = gnmt_model.GNMTModel
  else:
    raise ValueError("Unknown model architecture")

  train_model = create_train_model(model_creator, hparams, scope,
                                   single_cell_fn)
  eval_model = create_eval_model(model_creator, hparams, scope,
                                 single_cell_fn)
  # NOTE(qichao)
  if hparams.with_inference:
    infer_model = inference.create_infer_model(model_creator, hparams,
                                               scope, single_cell_fn)

  # Preload data for sample decoding.
  # dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
  # dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
  # sample_src_data = inference.load_data(dev_src_file)
  # sample_tgt_data = inference.load_data(dev_tgt_file)

  summary_name = "train_log"
  model_dir = hparams.out_dir

  # Log and output files
  log_file = os.path.join(out_dir, "log_%d" % time.time())
  log_f = tf.gfile.GFile(log_file, mode="a")
  utils.print_out("# log_file=%s" % log_file, log_f)

  avg_step_time = 0.0

  # TensorFlow model
  config_proto = utils.get_config_proto(
      log_device_placement=log_device_placement)

  train_sess = tf.Session(
      target=target_session, config=config_proto, graph=train_model.graph)
  eval_sess = tf.Session(
      target=target_session, config=config_proto, graph=eval_model.graph)
  if hparams.with_inference:
    infer_sess = tf.Session(
       target=target_session, config=config_proto, graph=infer_model.graph)

  with train_model.graph.as_default():
    loaded_train_model, global_step = model_helper.create_or_load_model(
        train_model.model, model_dir, train_sess, "train")

  # Summary writer
  summary_writer = tf.summary.FileWriter(
      os.path.join(out_dir, summary_name), train_model.graph)

  last_stats_step = global_step
  last_eval_step = global_step
  last_external_eval_step = global_step

  # This is the training loop.
  step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
  checkpoint_total_count = 0.0
  speed, train_ppl = 0.0, 0.0
  start_train_time = time.time()

  utils.print_out(
      "# Start step %d, lr %g, %s" %
      (global_step, loaded_train_model.learning_rate.eval(session=train_sess),
       time.ctime()),
      log_f)

  # Initialize all of the iterators
  skip_count = hparams.batch_size * hparams.epoch_step
  utils.print_out("# Init train iterator, skipping %d elements" % skip_count)
  train_sess.run(
      train_model.iterator.initializer,
      feed_dict={train_model.skip_count_placeholder: skip_count})

  while global_step < num_train_steps:
    ### Run a step ###
    start_time = time.time()
    try:
      # NOTE(qichao) train model
      step_result = loaded_train_model.train(train_sess)
      (_, step_loss, step_predict_count,
      step_summary, global_step, step_word_count, batch_size) = step_result
      hparams.epoch_step += 1
    except tf.errors.OutOfRangeError:
      # Finished going through the training dataset.  Go to next epoch.
      hparams.epoch_step = 0
      utils.print_out(
          "# Finished an epoch, step %d. Perform external evaluation" %
          global_step)
      train_sess.run(
          train_model.iterator.initializer,
          feed_dict={train_model.skip_count_placeholder: 0})
      continue

    # Write step summary.
    summary_writer.add_summary(step_summary, global_step)

    # update statistics
    step_time += (time.time() - start_time)

    # NOTE(qichao)
    checkpoint_loss += (step_loss * batch_size)
    checkpoint_predict_count += step_predict_count
    checkpoint_total_count += float(step_word_count)

    # Once in a while, we print statistics.
    if global_step - last_stats_step >= steps_per_stats:
      last_stats_step = global_step

      # Print statistics for the previous epoch.
      avg_step_time = step_time / steps_per_stats
      train_ppl = utils.safe_exp(checkpoint_loss / checkpoint_predict_count)
      speed = checkpoint_total_count / (1000 * step_time)
      utils.print_out(
          "  global step %d lr %g "
          "step-time %.2fs wps %.2fK step_loss %.6f ppl %.2f %s" %
          (global_step,
           loaded_train_model.learning_rate.eval(session=train_sess),
           avg_step_time, speed, step_loss, train_ppl, _get_best_results(hparams)),
          log_f)
      if math.isnan(train_ppl):
        break

      # Reset timer and loss.
      step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
      checkpoint_total_count = 0.0

    if global_step - last_eval_step >= steps_per_eval:
      last_eval_step = global_step

      utils.print_out("# Save eval, global step %d" % global_step)
      utils.add_summary(summary_writer, global_step, "train_ppl", train_ppl)
      run_internal_eval(eval_model, eval_sess, model_dir, hparams, summary_writer)

      # Save checkpoint
      loaded_train_model.saver.save(
          train_sess,
          os.path.join(out_dir, "translate.ckpt"),
          global_step=global_step)

    if global_step - last_external_eval_step >= steps_per_external_eval:
      last_external_eval_step = global_step

      # Save checkpoint
      loaded_train_model.saver.save(
          train_sess,
          os.path.join(out_dir, "translate.ckpt"),
          global_step=global_step)

  # Done training
  loaded_train_model.saver.save(
      train_sess,
      os.path.join(out_dir, "translate.ckpt"),
      global_step=global_step)

  utils.print_time("# Done training!", start_train_time)

  # run_eval_clk(eval_model, eval_sess, model_dir, hparams, summary_writer)
  summary_writer.close()

def _get_best_results(hparams):
  """Summary of the current best results."""
  tokens = []
  for metric in hparams.metrics:
    tokens.append("%s %.2f" % (metric, getattr(hparams, "best_" + metric)))
  return ", ".join(tokens)

def _internal_eval_clk(model, global_step, sess, iterator, iterator_feed_dict,
                       summary_writer, label, output_file):
  """Computing perplexity."""
  sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
  model_helper.output_hidden_state(model, sess, label, output_file)
