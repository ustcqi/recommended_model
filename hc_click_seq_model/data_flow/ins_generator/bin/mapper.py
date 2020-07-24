#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

for line in sys.stdin:
  line = line.strip()
  terms = line.split('\t')
  try:
    if terms[4] != "0":
      print terms[19].strip(), '\t', terms[4].strip(), '\t', terms[14].strip()
  except Exception, e:
    continue
