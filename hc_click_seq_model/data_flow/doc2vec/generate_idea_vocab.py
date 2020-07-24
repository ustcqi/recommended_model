#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

idea_title_srt_file = sys.argv[1]
idea_vocab_file = sys.argv[2]

def generate_idea_vocab(idea_title_file, idea_vocab_file):
  try:
    fd_idea_title = open(idea_title_file, "r")
    fd_idea_vocab = open(idea_vocab_file, "w+")
    for line in fd_idea_title.readlines():
      terms = line.strip().split("\t")
      idea_id = terms[0]
      fd_idea_vocab.write(idea_id + '\n')
  except Exception, e:
    print(e)

generate_idea_vocab(idea_title_srt_file, idea_vocab_file)
