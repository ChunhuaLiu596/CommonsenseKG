#!/bin/bash

#------------------
echo "#Creating alignment set"
conceptnet_source_file='./data/cn100k/cn100k_train_valid_test.txt'
swow_source_file='data/swow/SWOW-EN.R100.csv'
swow_filter_file='./data/swow/swow_triple_freq2.filter'

#<<'COMMENT'
output_folder=./data/alignment/C_S_V0.1

if [ ! -d "$output_folder" ]; then
    echo "Creating $output_folder"
    mkdir -p "$output_folder"
fi

align_dir="$output_folder"

##1. preprocess the swow file, filter by frequency
echo generate $swow_filter_file
python3 src/swow.py\
    --swow_file $swow_source_file\
    --wordpairs_frequency 2\
    --swow_freq_triple_file $swow_filter_file


#2. build net_cpn and net_swow, get_shared_nodes, shared_edges
echo align $conceptnet_source_file and $swow_filter_file
python src/align_cn_sw.py \
    --conceptnet_source_file $conceptnet_source_file\
    --swow_source_file $swow_filter_file\
    --align_dir $align_dir

#COMMENT

