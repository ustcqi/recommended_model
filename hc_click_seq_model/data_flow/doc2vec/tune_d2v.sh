source ./d2v.conf
grams_filename=./data/20180303/idea_grams.2018-03-04_08:15:48
idea_count=`cat $grams_filename | wc -l`
all_vector_txt=./test/vector.txt.test.9
all_model_txt=./test/model.txt.test.9
python train_doc2vec.py $grams_filename $idea_count $all_vector_txt $all_model_txt $iter $size $window $min_count
