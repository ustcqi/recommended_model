from __future__ import absolute_import, unicode_literals
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import collections
import datetime
import logging
import jieba
import jieba.posseg

idea_title_sorted_file = sys.argv[1]
grams_file = sys.argv[2]
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
grams_file = grams_file + '.' + timestamp

#jieba_cache_file = "./data/cache/"
jieba_tmp_dir = "./jieba/tmp/"

log_file = './log/split2token.log.' + timestamp
logging.basicConfig(filename=log_file, format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)
logging.info("running %s" % ' '.join(sys.argv))

pos_filt = frozenset(('ns', 'n', 'vn', 'v'))
STOP_WORDS = frozenset(('a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can',
                        'for', 'from', 'have', 'if', 'in', 'is', 'it', 'may',
                        'not', 'of', 'on', 'or', 'tbd', 'that', 'the', 'this',
                        'to', 'us', 'we', 'when', 'will', 'with', 'yet',
                        'you', 'your'))

def check_contain_chinese(check_str):
  #for ch in check_str.decode('utf-8'):
  for ch in check_str:
    if u'\u4e00' <= ch <= u'\u9fff':
      return True
  return False

def all_is_chinese(check_str):
  #for ch in check_str.decode('utf-8'):
  for ch in check_str:
    if u'\u4e00' > ch or ch > u'\u9fff':
      return False
  return True


#tokenizer = jieba.posseg.dt
jieba.dt.tmp_dir = jieba_tmp_dir
tokenizer = jieba.dt
#jieba.dt.cache_file = jieba_cache_file
total_num = 0
total_words = []
total_df = collections.defaultdict(int)
logging.info("open idea_title_sorted_file")
for line in open(idea_title_sorted_file, 'r'):
  total_num += 1
  pos = line.find('\t')
  if pos == -1:
    continue
  text = line[pos+1:].decode('utf-8')
  title_words = []
  for word in tokenizer.cut(text):
    #if all_is_chinese(word) and len(word.strip()) >=2 and word.lower() not in STOP_WORDS: 
    if all_is_chinese(word) and len(word.strip()) >=1 and word.lower() not in STOP_WORDS: 
      title_words.append(word)
  #for word,pos in tokenizer.cut(text):
  #  if all_is_chinese(word) and pos in pos_filt and len(word.strip()) >=2 and word.lower() not in STOP_WORDS: 
  #    out.append(word)
  total_words.append(title_words)
  df = set(title_words)
  for i in df:
    total_df[i] += 1

logging.info("Filtering words in total_words")
for idx in xrange(len(total_words)):
  new_words = []
  for w in total_words[idx]:
    if total_df[w]/1.0/total_num < 0.2:
      new_words.append(w)
  total_words[idx] = new_words

logging.info("Generating grams file")
grams_file_fd = open(grams_file, 'w+')
for word in total_words:
  grams_file_fd.write(' '.join(word) + '\n')
grams_file_fd.close()
logging.info("Grams file generated!")
print grams_file
