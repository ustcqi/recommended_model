#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import datetime
from operator import itemgetter

def load_idea_title_dict(idea_title_file):
  idea_title_dict = {}
  try:
    fd = open(idea_title_file)
  except Exception, e:
    raise
  else:
    for line in fd.readlines():
      line = line.strip()
      if not line:
        continue
      terms = line.split("\t")
      idea_id = terms[0].strip()
      title = terms[1].strip()
      idea_title_dict[idea_id] = title
    fd.close()
  return idea_title_dict

idea_title_file = sys.argv[1]
#idea_title_dict = load_idea_title_dict(idea_title_file)

user_idea_dict = {}

for line in sys.stdin:
  line = line.strip()
  terms = line.split('\t')
  try:
    key = terms[0].strip()
    fields = key.split(",")
    uid = fields[0].strip()
    date = fields[1].strip()
    idea_seq = terms[1].strip()
    ts_seq = terms[2].strip()
    #ideas = idea_seq.split('-')
    #for i in range(len(ts_seq)):
    #  if not idea_title_dict.has_key(ideas[i]):
    #    title = " "
    #  else:
    #    title = idea_title_dict[ideas[i]]
    #  title_seq += title + "-"
    #title_seq = title_seq[:-1]
    if not user_idea_dict.has_key(uid):
      # 每个 uid 会映射一个 三元组 (idea_seq, datetime_seq, title_seq)
      #user_idea_dict[uid] = (idea_seq, ts_seq, title_seq) 
      user_idea_dict[uid] = (idea_seq, ts_seq) 
    else:
      idea_s = user_idea_dict[uid][0] + '-' + idea_seq
      ts_s = user_idea_dict[uid][1] + '-' + ts_seq
      #title_s = user_idea_dict[uid][2] + '-' + title_seq
      #user_idea_dict[uid] = (idea_s, ts_s, title_s)
      user_idea_dict[uid] = (idea_s, ts_s)
  except Exception, e:
    continue


for uid in user_idea_dict.keys():
  try:
    #output = uid + "\t" + user_idea_dict[uid][0] + "\t" + user_idea_dict[uid][1] + "\t" + user_idea_dict[uid][2]
    output = uid + "\t" + user_idea_dict[uid][0] + "\t" + user_idea_dict[uid][1]
  except Exception, e:
    continue
  else:
    print(output)
