#!/bin/bash

#------------------
echo "#Testing alignment set"
conceptnet_source_file='./data/conceptnet/cn100k_train_valid_test.txt'
swow_source_file='./data/cn100k/cn100k_train_valid_test.txt'

output_folder=./data/alignment/C_S_V1.0

align_dir="$output_folder"

test_source_file= $align_folder+'/rel_triples_test'
valid_source_file= $align_folder+'/rel_triples_valid'
overlap_source_file= $align_folder+'/rel_triples_overlap12'

python3 src/test_alignment.py --concepenet_source_file $conceptnet_source_file\
--swow_source_file $swow_source_file\
--test_source_file $test_source_file\
--valid_source_file $valid_source_file\
--overlap_source_file $overlap_source_file\