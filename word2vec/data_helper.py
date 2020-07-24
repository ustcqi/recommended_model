# coding=utf-8
import sys
import 
import jieba

"""
basic data structure
1. word <-> index dict
2. word -> freq dict
3. word list(corpus, jieba cut)

traning samples
window=2
1. 掌握 深度 学习 需要 很强 的 数学 功底 
   => 
   x    y
   掌握 深度
   掌握 学习
   深度 掌握
   深度 学习
   深度 需要
   学习 掌握
   学习 深度
   学习 需要
   学习 很强
   需要 深度
   需要 学习
   需要 很强
   需要 的
   ...
"""
class DataHelper(object):
 
  def __init__(self):
    pass

  
