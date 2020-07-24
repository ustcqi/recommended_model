#!/bin/bash
set -x
file=`readlink -f $0`
script_path=`dirname $file`
cd $script_path
source ./d2v.conf
source ../../global.conf
yesterday=$1
date_dir=./data/$yesterday
all_start_time=`date +%s`
idea_txt=$date_dir/idea.txt
if [ ! -f $idea_txt ];then
  echo "idea.txt doesn't exits."
  exit 1
fi
idea_title_file=$date_dir/idea_title.dat
ret=`python get_acccount_title.py $idea_txt $idea_title_file`
if [ $? -eq 0 ];then
  echo "python get_account_title.py $idea_txt $idea_title_file ok."
else
  echo "python get_account_title.py $idea_txt $idea_title_file failed."
  exit 1
fi
idea_title_sorted_file=$ret.sorted
sort $ret -k1 -g > $idea_title_sorted_file
if [ $? -eq 0 ];then
  echo "sort $ret -k1 -g > $idea_title_sorted_file ok."
else
  echo "sort $ret -k1 -g > $idea_title_sorted_file failed."
  exit 1
fi
start_time=`date +%s`
grams_file=$date_dir/idea_grams
grams_filename=`python split2token.py $idea_title_sorted_file $grams_file`
if [ $? -eq 0 ];then
  end_time=`date +%s`
  cost_time=$[ end_time - begin_time ]
  echo "python split2token.py $idea_title_sorted_file $grams_filename ok, cost $cost_time seconds."
else
  echo "python split2token.py $idea_title_sorted_file $grams_filename failed."
  exit 1
fi
all_model_txt=$date_dir/model.txt
all_vector_txt=$date_dir/vector.txt
idea_count=`cat $grams_filename | wc -l`
# check successful
echo "Generate grams file successfully"
start_time=`date +%s`
python train_doc2vec.py $grams_filename $idea_count $all_vector_txt $all_model_txt $iter $size $window $min_count
if [ $? -eq 0 ];then
  end_time=`date +%s`
  cost_time=$[ end_time - begin_time ]
  echo "train doc2vec successful, cost $cost_time seconds."
  echo "padding doc2vec..."
  python pad_doc2vec.py $all_vector_txt $padding_lines $size $padding_value
else
  echo "train doc2vec failed."
  exit 1
fi
echo "generate vocab.idea"
python generate_idea_vocab.py $idea_title_sorted_file $date_dir/$vocab_idea_name
if [ $? -ne 0 ];then
  echo "generate vocab.idea failed."
  exit 1
fi
#cp -f $data_dir/vocab.idea /tmp/s2s_data/vocab.idea
all_end_time=`date +%s`
all_cost_time=$[ all_end_time - all_start_time ]
echo "doc2vec.sh cost $all_cost_time seconds." > ./data/$yesterday/doc2vec.donefile
exit 0
