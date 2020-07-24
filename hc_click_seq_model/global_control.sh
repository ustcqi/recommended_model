#!/bin/bash
source ./global.conf
set -x
file=`readlink -f $0`
script_path=`dirname $file`
cd $script_path

today=`date +%Y%m%d`
yesterday=`date -d yesterday +%Y%m%d`
cd $data_flow_dir
sh -x data_flow.sh $today > log/$today.log
if [ $? -ne 0 ];then
  echo "generate seq2seq samples failed."
  exit 1
fi

cd $script_path
if [ ! -f $data_flow_dir/ins_generator/data/$yesterday/train.idea ];then
  echo "train data $train_idea_name not exists"
  exit 1
fi
mkdir $nmt_dir/data/$yesterday
mkdir $nmt_dir/models/$yesterday
# move all the requried data needed for nmt, such as doc2vec vector, vocab.idea, train.idea
# copy doc2vec as the nmt embedding
cp $data_flow_dir/doc2vec/data/$yesterday/vector.txt $nmt_dir/data/$yesterday/doc2vec_emb
# copy vocab.idea as the nmt vocab file
cp $data_flow_dir/doc2vec/data/$yesterday/vocab.idea $nmt_dir/models/$yesterday
# move the training data
if [ ! -f $data_flow_dir/ins_generator/data/$yesterday/train.idea ];then
  echo "training data not exists"
  exit 1
fi
mv $data_flow_dir/ins_generator/data/$yesterday/train.idea $nmt_dir/data/$yesterday/
# generate data from the training data for evaluation
sample_count=`cat $nmt_dir/data/$yesterday/train.idea | wc -l`
valid_count=$(echo "$sample_count * 0.2"|bc)
valid_count=`printf "%.f" $valid_count`
head -$valid_count $nmt_dir/data/$yesterday/train.idea > $nmt_dir/data/$yesterday/test.idea
tail -n +$valid_count $nmt_dir/data/$yesterday/train.idea > $nmt_dir/data/$yesterday/train.idea.bak
# training data
mv $nmt_dir/data/$yesterday/train.idea.bak $nmt_dir/data/$yesterday/train.idea

cd $script_path/$nmt_dir
mkdir ./log/$today
let No=`ls ./log/$today | sort -gr | head -1`
if [ "$No" = "" ];then
  sh -x run.sh $yesterday 1 > ./log/$today/1.log
else
  let No=`expr $No + 1`  
  sh -x run.sh $yesterday $No > ./log/$today/$No.log
fi
exit 0
