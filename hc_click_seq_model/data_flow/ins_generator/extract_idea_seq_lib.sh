#!/bin/bash
source ./ins.conf

# 抽取一天的 idea 序列
function extract_idea_seq_day() {
  local input=$1
  local output=$2
  $HADOOP fs -test -e $input
  if [ $? -ne 0 ];then
    echo "Failed:Input path doesn't exit of function extract_idea_seq_day()"
    return 1 
  fi
  $HADOOP fs -test -e $output/_SUCCESS
  if [ $? -eq 0 ];then
    echo "Output directory already exisit and its stat is success."
    return 0
  fi
  $HADOOP jar $HADOOP_BASE/contrib/streaming/hadoop-streaming-1.2.0.jar \
    -file "./bin/mapper.py" \
    -file "./bin/reducer.py" \
    -input $input \
    -output $output \
    -mapper "./mapper.py" \
    -reducer "./reducer.py" \
    -jobconf stream.num.map.output.key.fields=1 \
    -jobconf stream.num.reduce.output.key.fields=1 \
    -jobconf mapreduce.job.running.map.limit=2000 \
    -jobconf mapreduce.job.running.reduce.limit=2000 \
    -jobconf mapreduce.map.memory.mb=10000 \
    -jobconf mapreduce.job.maps=500 \
    -jobconf mapreduce.job.reduces=300 \
    -jobconf mapreduce.input.fileinputformat.split.minsize=1000000000 \
    -jobconf mapreduce.job.name="clk_idea_generator" \
    -jobconf mapred.job.priority=NORMAL
}

# 合并新的 idea 序列
function merge_new_idea_seq() {
  local input=$1
  local output=$2
  yesterday=$3
  $HADOOP fs -test -e $output/_SUCCESS
  if [ $? -eq 0 ];then
    echo "Output directory already exisit and its stat is success."
    return 0
  fi
  $HADOOP fs -rm -r $output

  $HADOOP jar /home/serving/hadoop_client/hadoop/contrib/streaming/hadoop-streaming-1.2.0.jar \
    -file "./bin/all_mapper.py" \
    -file "./bin/all_reducer_second_sort.py" \
    -file "./data/$yesterday/idea_title_dict" \
    -input $input \
    -output $output \
    -mapper "./all_mapper.py" \
    -reducer "./all_reducer_second_sort.py idea_title_dict" \
    -jobconf mapreduce.job.running.map.limit=2000 \
    -jobconf mapreduce.job.running.reduce.limit=2000 \
    -jobconf mapreduce.map.memory.mb=10000 \
    -jobconf mapreduce.job.maps=300 \
    -jobconf mapreduce.job.reduces=300 \
    -jobconf mapreduce.input.fileinputformat.split.minsize=1000000000 \
    -jobconf mapreduce.job.name="clk_title_seq_generator" \
    -jobconf stream.map.output.field.separator=\t \
    -jobconf stream.num.map.output.key.fields=2 \
    -jobconf mapreduce.map.output.key.field.separator=, \
    -jobconf num.key.fields.for.partition=1 \
    -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
    -jobconf mapreduce.job.priority=NORMAL
}

# 生成最终 idea 序列样本
function extract_seq2seq_instance() {
  input=$1
  output=$2
  window=$3
  x_stride=$4
  y_stride=$5
  min_len=$6
  $HADOOP fs -test -e $input
  if [ $? -ne 0 ];then
    echo "Failed:Input path doesn't exit of function extract_seq2seq_instance()"
    return 1 
  fi
  $HADOOP fs -test -e $output/_SUCCESS
  if [ $? -eq 0 ];then
    echo "Output directory already exisit and its stat is success."
    return 0
  fi
  $HADOOP fs -rm -r $output
  $HADOOP jar /home/serving/hadoop_client/hadoop/contrib/streaming/hadoop-streaming-1.2.0.jar \
    -file "./bin/ins_reducer_pad.py" \
    -input $input \
    -output $output \
    -mapper "./ins_reducer_pad.py $window $x_stride $y_stride $min_len" \
    -jobconf stream.num.map.output.key.fields=1 \
    -jobconf stream.num.reduce.output.key.fields=1 \
    -jobconf mapreduce.job.running.map.limit=2000 \
    -jobconf mapreduce.job.running.reduce.limit=2000 \
    -jobconf mapreduce.map.memory.mb=10000 \
    -jobconf mapreduce.job.maps=500 \
    -jobconf mapreduce.job.reduces=300 \
    -jobconf mapreduce.input.fileinputformat.split.minsize=1000000000 \
    -jobconf mapreduce.job.name="idea_seq_ins_generator" \
    -jobconf mapred.job.priority=NORMAL
}
