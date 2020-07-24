#!/bin/bash
set -x
source ./ins.conf
source ./extract_idea_seq_lib.sh

extract_day=$1
last_merged_day=$2
last_merged_output=$3
merge_output=$4
merge_input=""
yesterday=$extract_day

# 抽取上次 merge 后的每天 idea 序列
earliest_day=`date +%Y%m%d -d "$extract_ins_days day ago $extract_day"`
while [ $extract_day -ge $earliest_day ]
do
  input=$input_base/dt=$extract_day
  output=$day_idea_seq_base/$extract_day
  extract_idea_seq_day $input $output
  # 已生成好或hadoop任务执行成功
  if [ $? -eq 0 ];then
    extract_day=`date +%Y%m%d -d "1 day ago $extract_day"`
    merge_input="${output}/part-*,$merge_input"
    continue
  fi
  extract_day=`date +%Y%m%d -d "1 day ago $extract_day"`
  merge_input=$output/part-*,$merge_input
done
echo "new idea sequences have been generated."

merge_input=${merge_input%?}
merge_new_idea_seq $merge_input $merge_output $yesterday
if [ $? -ne 0 ];then
  echo "input doesn't exit or mapreduce job failed."
fi
