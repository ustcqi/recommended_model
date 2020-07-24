#!/usr/bin/env python
import sys

window = int(sys.argv[1])
x_stride = int(sys.argv[2])
y_stride = int(sys.argv[3])
min_length = int(sys.argv[4])

def generate_padding(idea_seq, window):
  for i in range(window-1):
    idea_seq = '0-' + idea_seq + '-0'
  return idea_seq

def findnth(original, obj, n):
  if n == 0:
    return -1
  parts = original.split(obj, n)
  if len(parts) < (n + 1):
    return -1
  return len(original) - len(parts[-1]) - len(obj)

def generate_mask(instance):
  idea_seq = instance.split('-')
  mask = ''
  for idea in idea_seq:
    if idea == '0':
      mask = mask + '0' + '-'
    else:
      mask = mask + '1' + '-'
  return mask[:-1]

def instance_filter(x):
  instance = ""
  try:
    xs = x.split('-')
    for x in xs:
      if x != '0':
        instance += x + '-'
    instance = instance[:-1]
  except Exception, e:
    pass
  return instance

user_ins_dict = {}

user_instances = {}
for line in sys.stdin:
  instances = []
  x_masks = []
  y_masks = []
  line = line.strip()
  terms = line.split("\t")
  try:
    uid = terms[0]
    idea_seq = terms[1].strip()
    idea_seq = generate_padding(idea_seq, window)
    idea_seq = idea_seq + '-'
    seq_len = len(idea_seq.split('-'))
    if  seq_len < min_length:
      continue
  except Exception, e:
    continue
  else:
    i = 0
    while (i + y_stride + window) < seq_len:
      xs = findnth(idea_seq, "-", i) + 1
      xe = findnth(idea_seq, "-", i+window)
      x = idea_seq[xs : xe]  
      ys = findnth(idea_seq, "-", i+y_stride) + 1
      ye = findnth(idea_seq, "-", i+y_stride+window)
      y = idea_seq[ys : ye]
      # instance = x + "\t" + y
      instance = instance_filter(x) + '\t' + instance_filter(y) 
      # instance filter
      instances.append(instance)
      x_mask = generate_mask(x)
      y_mask = generate_mask(y)
      x_masks.append(x_mask)
      y_masks.append(y_mask)
      i += x_stride
    user_instances[uid] = instances
    # output instances
    for i in range(len(instances)):
      output = uid + '\t' + instances[i] + '\t' + x_masks[i] + '\t' + y_masks[i]
      print output
