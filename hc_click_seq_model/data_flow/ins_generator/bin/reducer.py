#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import datetime
from operator import itemgetter

user_idea_dict = {}

for line in sys.stdin:
  line = line.strip()
  terms = line.split('\t')
  try:
    uid = terms[0].strip()
    timestamp = terms[1].strip()
    #ts = float(timestamp[:-6])
    #dt = datetime.datetime.fromtimestamp(ts)
    #dts = dt.strftime("%Y/%m/%d %H:%M:%S")
    idea_id = terms[2].strip()
  except Exception, e:
    continue
  if not user_idea_dict.has_key(uid):
    time_idea_dict = {}
    time_idea_dict[timestamp] = idea_id
    user_idea_dict[uid] = time_idea_dict
  else:
    user_idea_dict[uid][timestamp] = idea_id

for uid in user_idea_dict.keys():
  output = uid
  time_idea_dict = user_idea_dict[uid]
  idea_seq = ""
  ts_seq = ""
  for timestamp in sorted(time_idea_dict.keys()):
    idea_seq += time_idea_dict[timestamp] + "-"
    ts_seq += timestamp + "-"
  output += "\t" + idea_seq[:-1] + "\t" + ts_seq[:-1]
  print output
