#!/bin/bash
set -x
source ./ins.conf
source ./extract_idea_seq_lib.sh

today=$1
yesterday=`date +%Y%m%d -d "1 day ago $today"`

# 下载上一次 merge 后的 donefile
if [ -f "./donefile.srt" ];then
  rm -f donefile.srt
fi
$HADOOP fs -get $donefiles/donefile.srt ./
if [ $? -ne 0 ];then
  echo "download donefile.srt failed."
  # 若之前没有合并过, 则上次 merge 日期为 yesterday 之前的 extract_ins_days
  last_merged_day=`date +%Y%m%d -d "$extract_ins_days day ago $yesterday"`
  last_merged_output="no_merge"
else
  source ./donefile.srt
  last_merged_day=$merged_day
  last_merged_output=$merged_output_path
  $HADOOP fs -rm $donefiles/donefile.srt
fi
merge_output=$merged_idea_seq_base/$today
extract_day=$yesterday

# 对新的 idea sequence 进行合并
sh -x merge_idea_seq_srt.sh $extract_day $last_merged_day $last_merged_output $merge_output
if [ $? -ne 0 ];then
  echo "merge_idea_seq_srt.sh error."
  exit 1
fi
rm -f ./donefile.srt
echo "merged_output_path=$merge_output" > donefile.srt
echo "merged_day=$today" >> donefile.srt
$HADOOP fs -put donefile.srt $donefiles
if [ $? -ne 0 ];then
  echo "put donefile.srt failed."
  exit 1
fi
rm -f donefile.srt

# 生成最终的 seq2seq 样本
extract_seq2seq_instance $merge_output $ins_base/$today $window $x_stride $y_stride $min_length
if [ $? -ne 0 ];then
  echo "input doesn't exits or generate final instances failed."
  exit 1
fi
mkdir ./data/$yesterday
$HADOOP fs -getmerge $ins_base/$today ./data/$yesterday/train.idea
if [ $? -ne 0 ];then
  echo "merge final samples error."
  exit 1
fi
