#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

def load_idea_title_dict(filename):
  idea_title_dict = {}
  try:
    fd = open(filename)
  except Exception, e:
    raise
  else:
    for line in fd.readlines():
      line = line.strip()
      terms = line.split("\t")
      idea_id = terms[0].strip()
      title = terms[9].strip()
      if not title:
        continue
      idea_title_dict[idea_id] = title
    fd.close()
  return idea_title_dict


def generate_idea_title_dict(filename, idea_title_dict):
  try:
    fd = open(filename, 'wb+')
  except Exception, e:
    raise
  else:
    for idea_id in idea_title_dict.keys():
      line = idea_id + "\t" + idea_title_dict[idea_id] + "\n"
      fd.write(line)
    fd.close()
  
def main(idea_txt_file, idea_title_file):
  idea_title_dict = load_idea_title_dict(idea_txt_file)
  generate_idea_title_dict(idea_title_file, idea_title_dict)  

idea_txt_file = sys.argv[1]
idea_title_file = sys.argv[2]

main(idea_txt_file, idea_title_file)
