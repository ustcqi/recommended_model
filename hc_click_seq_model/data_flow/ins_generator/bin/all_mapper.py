#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import datetime

input_file = os.environ["map_input_file"]


for line in sys.stdin:
  line = line.strip()
  terms = line.split("\t")
  try:
    uid = terms[0].strip()
    idea_seq = terms[1].strip()
    ts_seq = terms[2].strip()
    ss = input_file.split('/')
    date = ss[len(ss) - 2]
    # convert string to datetime
    dt = datetime.datetime.strptime(date, '%Y%m%d')
    # 保证上一次合并的日期在新的日期之前
    if str(input_file).find('merge') != -1:
      dt = dt - datetime.timedelta(days=1) 
    # convert datetime to string
    date = dt.strftime('%Y%m%d')
  except Exception, e:
    continue
  else:
    output = uid + "," + date + "\t" + idea_seq + "\t" + ts_seq
    print output
