#!/usr/bin/bash
out_put_dir=$1
cd "${out_put_dir}" || exit
rm -rf partitioned_train_data
mkdir partitioned_train_data
#shuf "_train.rating" > "_train.rating_shuffled"
split -l 256000 -a 3 -d "_train.rating" partitioned_train_data/part_
#split -l 256000 -a 3 -d "_train.rating_shuffled" partitioned_train_data/part_