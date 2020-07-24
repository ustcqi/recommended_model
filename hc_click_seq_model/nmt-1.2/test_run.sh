#!/bin/bash
s2s_data_base=/app/user/qichao/ali/seq_model/seq2seq/nmt-1.2/data
s2s_model_base=/app/user/qichao/ali/seq_model/seq2seq/nmt-1.2/models
serial_num=model_5

python -m nmt.nmt \
    --src=idea \
    --tgt=idea \
    --vocab_prefix=$s2s_model_base/vocab  \
    --share_vocab=True \
    --train_prefix=$s2s_data_base/train \
    --test_prefix=$s2s_data_base/test \
    --out_dir=$s2s_model_base/$serial_num \
    --time_major=False \
    --load_embedding=True \
    --doc2vec_emb=$s2s_data_base/doc2vec_emb \
    --emb_dim=64 \
    --src_reverse=True \
    --src_max_len=5 \
    --tgt_max_len=5 \
    --hidden_state_file=$s2s_model_base/$model_name/hidden_state \
    --with_output_layer=False \
    --with_inference=True \
    --num_buckets=1 \
    --source_reverse=False \
    --loss_type=square_loss \
    --num_train_steps=62500000 \
    --attention=bahdanau \
    --attention_architecture=standard \
    --steps_per_stats=1000 \
    --learning_rate=0.0005 \
    --num_layers=2 \
    --num_units=64 \
    --batch_size=64 \
    --dropout=0.2 \
    --metrics=bleu
