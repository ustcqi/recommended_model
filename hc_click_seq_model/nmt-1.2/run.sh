#!/bin/bash
source ../global.conf
source ./fixed_hparams.conf
source ./tuning_hparams.conf
day=$1
model_dir=$2
file=`readlink -f $0`
script_path=`dirname $file`
sample_count=`cat data/$day/train.idea | wc -l`
let num_train_steps=$(echo "$sample_count / $batch_size"|bc)
echo $num_train_steps

mkdir $script_path/models/$day
mkdir $script_path/data/$day
s2s_model_base=$script_path/models/$day
s2s_data_base=$script_path/data/$day
mkdir $s2s_model_base/$model_dir
# check if the required files exists
if [ ! -f $s2s_data_base/$doc2vec_name ];then
  echo "doc2vec not exisis"
  exit 1
fi

if [ ! -f $s2s_data_base/$train_data ];then
  echo "nmt train data not exists"
  exit 1
fi

if [ ! -f $s2s_data_base/$test_data ];then
  echo "nmt test data not exists"
  exit 1
fi

if [ ! -f $s2s_model_base/$vocab_idea_name ];then
  echo "nmt $vocab_idea_name not exists"
  exit 1
fi

python -m nmt.nmt \
    --src=$src \
    --tgt=$tgt \
    --vocab_prefix=$s2s_model_base/vocab  \
    --share_vocab=$share_vocab \
    --train_prefix=$s2s_data_base/train \
    --test_prefix=$s2s_data_base/test \
    --out_dir=$s2s_model_base/$model_dir \
    --time_major=$time_major \
    --load_embedding=$load_embedding \
    --doc2vec_emb=$s2s_data_base/$doc2vec_name \
    --emb_trainable=$emb_trainable \
    --emb_dim=$emb_dim \
    --src_embed_size=$src_embed_size \
    --tgt_embed_size=$tgt_embed_size \
    --src_reverse=$src_reverse \
    --src_max_len=$src_max_len \
    --tgt_max_len=$tgt_max_len \
    --hidden_state_file=$s2s_model_base/hidden_state \
    --with_output_layer=$with_output_layer \
    --with_inference=$with_inference \
    --num_buckets=$num_buckets \
    --source_reverse=$source_reverse \
    --loss_type=$loss_type \
    --num_train_steps=$num_train_steps \
    --attention=$attention \
    --attention_architecture=$attention_architecture \
    --steps_per_stats=$steps_per_stats \
    --learning_rate=$learning_rate \
    --num_layers=$num_layers \
    --num_units=$num_units \
    --batch_size=$batch_size \
    --dropout=$dropout \
    --metrics=$metrics
