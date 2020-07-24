import sys
import datetime
import collections

idea_filename = sys.argv[1]
idea_title_filename = sys.argv[2]
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

acc_dict = collections.defaultdict(set)
for line in open(idea_filename, 'r'):
  line = line.strip().split('\t')
  if len(line) < 10:
    continue
  #acc = line[1]
  acc = line[0]
  title = line[9].strip()
  if title:
    acc_dict[acc].add(title)
  else:
    pass

idea_title_filename = idea_title_filename + '.' + timestamp
ACCOUNT_FILE = open(idea_title_filename, 'w+')
for i in acc_dict:
  out = ' '.join(list(acc_dict[i]))
  ACCOUNT_FILE.write(i + "\t" + out + '\n')
print idea_title_filename
