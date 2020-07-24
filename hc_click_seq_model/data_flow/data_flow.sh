#!/bin/bash
source ../global.conf
set -x
file=`readlink -f $0`
script_path=`dirname $file`
cd $script_path

all_start_time=`date +%s`
if [ $# -gt 0 ];then
  today=$1
else
  # for local test
  today=`date +%Y%m%d`
fi
#today=$1
yesterday=`date +%Y%m%d -d "1 day ago $today"`
cur_dir=`pwd`

# copy idea.txt from 11.251.203.177
if [ ! -f "$cur_dir/doc2vec/data/$yesterday/idea.txt" ];then
  mkdir $cur_dir/doc2vec/data/$yesterday
  scp serving@11.251.203.177:/serving/db_dump/$yesterday/huichuan_ad/idea.txt $cur_dir/doc2vec/data/$yesterday/idea.txt
  if [ $? -eq 0 ];then
    echo "idea.txt has been copied."
  else
    echo "copy idea.txt failed."
    exit 1
  fi
fi

# train doc2vec
cd ./doc2vec
sh -x doc2vec.sh $yesterday > doc2vec.${yesterday}.log
if [ $? -ne 0 ];then
  echo "doc2vec training failed."
  exit 1
fi

# generate idea:title dict 
cd ../
mkdir $cur_dir/ins_generator/data/$yesterday
if [ ! -f "$cur_dir/ins_generator/data/$yesterday/idea_title_dict" ];then
  python generate_idea_title_dict.py $cur_dir/doc2vec/data/$yesterday/idea.txt \
                                   $cur_dir/ins_generator/data/$yesterday/idea_title_dict
  if [ ! -f "./ins_generator/data/$yesterday/idea_title_dict" ];then
    echo "generate idea:title dict failed."
    exit 1
  fi
fi

# extract idea sequence and generate seq2seq instance
cd ./ins_generator
sh -x instance_flow.sh $today
if [ $? -ne 0 ];then
  echo "generate final idea sequence samples error."
  exit 1
fi

all_end_time=`date +%s`
all_cost_time=$[ all_end_time - all_start_time ]
echo "data_flow.sh cost $all_cost_time seconds."
exit 0
