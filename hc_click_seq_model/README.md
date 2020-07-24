####################################汇川点击序列模型文档#####################################
#***       Author: qichao         ***#
#***      Version: v_1            ***#
#***         Date: 2018/03/09     ***#
#***  Description:                ***#
#############################################################################################

每个模块的文档 README 在各自目录下: 
  nmt-1.2/readme, data_flow/README, data_flow/doc2vec/README, data_flow/ins_generator/README

一. 使用方法
  1. 进入 hc_click_seq_model 目录
  2. sh -x global_control.sh / nohup sh -x global_control.sh > log/1.log 2>&1 &
  3. 运行完后, 会在 nmt-1.2/models 目录生成模型的结果, models 目录中会包含多个日期的子目录,
     子目录内是当天训练的模型的结果. 里面包含模型的超参数文件 hparams 和 用户恢复模型的 checkpoint文件.
     在 nmt-1.2/data 目录中也包含多个模型运行的日期的目录, 里面包含训练数据 train.idea 和 测试数据test.idea .
     也包含 doc2vec 训练生成的 doc2vec_emb 文件. 以及最新的 idea 集合文件 vocab.idea .
  4. 整个数据流中不包括 doc2vec 和 nmt 模型的调参, 因此在调参阶段, 需分别在各自目录调参数配置再执行.

二. 数据流程
  1. 进入 data_flow, 执行 data_flow.sh 用于生成最终的样本, 
     过程分为三个阶段:
       1) 解析每天的日志数据, 生成 uid \t idea_sequence \t timestamp_sequence格式的数据.
       2) 按照 uid 对近 30 天的数据合并并按照 timestamp 排序, 生成中间样本.
       3) 对 2) 中的中间样本进行切分, 生成最终的 seq2seq 格式的样本.
     其中 ins.conf 是相关的参数配置, 可以修改.
  2. 进入 doc2vec, 执行 doc2vec.sh 用于对 idea 级别的 doc2vec 训练, 生成每个 idea title 对应的do2vec 结果.
  3. 生成 vocab.idea 的 idea 词表, 用于 nmt 的训练.
  4. 样本准备好并 doc2vec 训练完成后, 进入 nmt-1.2 执行 run.sh , 里面有 fixed_hparams.conf 和 tuning_hparams.conf
     两个超参数配置文件, 其中 fixed_hparams.conf 里的参数通常不需要调整, tuning_hparams.conf 用于调整模型参数.

三. 模型结构
  1. 基于原始的 nmt-1.2 修改, 最主要的修改包括以下几个部分:
     1) 修改了数据生成器模块 iterator_utils.py, 使其根据我们的样本格式生成训练数据.
     2) decoder 阶段多了一路序列重建, 在 model.py 中添加了 def _build_rebuild_decoder(...) 函数
        以及 def _build_graph(...) 函数
     3) 在 train.py 中修改了 def train(...) 函数, 以及评估函数的修改.
     4) model.py 中 loss function 的修改

四. 目录结构 及 文件说明
.
├── data_flow (准备样本的数据流目录)
│ ├── data_flow.sh (样本生成的 执行脚本)
│ ├── doc2vec (doc2vec 目录)
│ │ ├── d2v.conf (doc2vec 的参数配置文件)
│ │ ├── data (生成的数据结果及模型结果目录)
│ │ │ └── 20180301 (当天最新的 idea.txt 生成的 doc2vec 结果及 vocab.idea)
│ │ │     ├── doc2vec.donefile
│ │ │     ├── idea_grams.2018-03-02_13:56:38
│ │ │     ├── idea_title.dat.2018-03-02_13:55:44
│ │ │     ├── idea_title.dat.2018-03-02_13:55:44.sorted
│ │ │     ├── idea.txt
│ │ │     ├── model.txt.2018-03-02_11:19:08
│ │ │     ├── model.txt.2018-03-02_11:19:08.docvecs.doctag_syn0.npy
│ │ │     └── vocab.idea
│ │ ├── doc2vec.20180301.log (doc2vec 训练的日志)
│ │ ├── doc2vec.sh (doc2vec 训练数据流)
│ │ ├── generate_idea_vocab.py (生成 vocab.idea 的脚本)
│ │ ├── get_acccount_title.py  (doc2vec 训练辅助脚本)
│ │ ├── jieba
│ │ │ └── tmp
│ │ │     └── jieba.cache
│ │ ├── log
│ │ │ ├── doc2vec.log.2018-03-02_14:29:00
│ │ │ └── split2token.log.2018-03-02_13:56:38
│ │ ├── pad_doc2vec.py (用于在训练生成 doc2vec 文件中在前面几行补 0)
│ │ ├── split2token.py (doc2vec 训练辅助脚本)
│ │ └── train_doc2vec.py (doc2vec 训练脚本)
│ ├── generate_idea_title_dict.py (辅助脚本)
│ ├── ins_generator (样本生成总目录)
│ │ ├── bin (hadoop 任务需要的脚本文件夹)
│ │ │ ├── all_mapper.py  (中间样本生成的 mapper 脚本)
│ │ │ ├── all_reducer_second_sort.py  (中间样本生成的 reducer 脚本)
│ │ │ ├── ins_reducer_pad.py (最终样本生成的 reducer 脚本)
│ │ │ ├── mapper.py  (每天日志解析的 mapper 脚本)
│ │ │ └── reducer.py  (每天日志解析的 reducer 脚本)
│ │ ├── data
│ │ │ └── 20180301
│ │ │     └── idea_title_dict
│ │ ├── extract_idea_seq_lib.sh  (样本生成的 hadoop 任务的相关函数)
│ │ ├── ins.conf  (样本生成的配置文件)
│ │ ├── instance_flow.sh  (样本生成的执行脚本)
│ │ ├── merge_idea_seq_srt.sh  (样本生成的核心脚本)
│ ├── log
│ │ └── 20180302.log
│ └── README
├── general_control.sh  (整个数据流的执行脚本)
├── global.conf  (配置文件)
├── log  (日志目录)
│ └── 20180301.log
├── nmt-1.2  (模型目录)
│ ├── CONTRIBUTING.md
│ ├── data  (模型训练需要的数据目录)
│ │ ├── 20180301
│ │ │ ├── doc2vec_emb  (doc2vec 结果, 用于做 nmt 的 embedding)
│ │ │ ├── test.idea  (测试数据)
│ │ │ └── train.idea  (训练数据)
│ ├── LICENSE
│ ├── log
│ │ └── 20180302
│ ├── models  (nmt 训练的结果)
│ │ └── 20180303
│ │     ├── best_bleu
│ │     ├── hparams
│ ├── nmt
│ │ ├── attention_model.py
│ │ ├── gnmt_model.py
│ │ ├── inference.py
│ │ ├── inference_test.py
│ │ ├── __init__.py
│ │ ├── model_helper.py
│ │ ├── model.py
│ │ ├── model_test.py
│ │ ├── nmt.py
│ │ ├── train.py
│ │ └── utils
│ │     ├── common_test_utils.py
│ │     ├── evaluation_utils.py
│ │     ├── evaluation_utils_test.py
│ │     ├── __init__.py
│ │     ├── iterator_utils.py
│ │     ├── iterator_utils_test.py
│ │     ├── misc_utils.py
│ │     ├── misc_utils_test.py
│ │     ├── nmt_utils.py
│ │     ├── vocab_utils.py
│ │     └── vocab_utils_test.py
│ ├── README.md
│ ├── run.sh  (nmt 模型训练脚本)
│ └── test_run.sh
└── README
