#!/bin/bash

# Get conceptnet triples
#python3 src/preprocess.py conceptnet

# Preprocess dataset
#python3 src/preprocess.py

# Start training
#python3 -u src/main.py --gpu 0 > run.log &


#nohup python3 src/swow-conceptnet.py --process_conceptnet --conceptnet_tuple_file ./data/concept.filter.v1 --conceptnet_weight_threshold 1.0 2>log/f21 1>log/f1_weigt1.0 &
#nohup python3 src/swow-conceptnet.py --process_conceptnet --conceptnet_tuple_file ./data/concept.filter.v2 --conceptnet_weight_threshold 0.0 2>log/f22 1>log/f1_weigt0.0 &

  #--process_conceptnet\
  #--conceptnet_weight_threshold 0.0\
  #--conceptnet_raw_file /home/chunhua/Commonsense/basic-data/conceptnet-assertions-5.7.0.csv\
    #--diffvec_tuple_file  /home/chunhua/Commonsense/basic-data/WordRelation/word_pairs_final.SEMBLESS.csv\

#python3 src/swow-conceptnet.py\
#    --swow_file /home/chunhua/Commonsense/SWOWEN-2018/data/2018/SWOW-EN.R100.csv\
#    --fuzzy_match\
#    --num_neigbbour_hops 2\
#    --wordpairs_frequency 2


#------------------
swow_triple_file="./data/swow/swow_triple_freq2.filter"
shared_vocab_file='data/swow/conceptnet_swow_vocab.shared'
conceptnet_triple_file='data/conceptnet/cn-100k-hrt.txt'


#1. preprocess the swow file, filter by frequency
echo "Step1: preprocess swow file ..."
 python3 src/swow.py\
    --swow_file /home/chunhua/Commonsense/SWOWEN-2018/data/2018/SWOW-EN.R100.csv\
    --swow_triple_file $swow_triple_file\
    --wordpairs_frequency 2

#2. build net_cpn and net_swow, get_shared_nodes, shared_edges
echo "Step2: generate shared nodes, edges ..."
python src/swow-conceptnet.py \
    --conceptnet_quadruple $conceptnet_triple_file\
    --swow_triple $swow_triple_file\
    --shared_vocab_file $shared_vocab_file
    #1--lemma_vocab


#3. split swow into train/valid/test
#python src/swow_split.py --input_file $swow_triple_file --shared_vocab $shared_vocab_file






###########if use ConceptNet5.7.0 ###########
#1. preprocess the swow file to
 #python3 src/swow.py\
 #   --swow_file /home/chunhua/Commonsense/SWOWEN-2018/data/2018/SWOW-EN.R100.csv\
 #   --swow_triple_file ./data/swow_triple_frequenc2_0420\
 #   --wordpairs_frequency 2


#2. preprocess the conceptnet file to
#python3 src/conceptnet.py --process_conceptnet\
#    --conceptnet_tuple_file ./data/concept.filter\
#    --conceptnet_weight_threshold 1.0\
#    --conceptnet_raw_file /home/chunhua/Commonsense/basic-data/conceptnet-assertions-5.7.0.csv\


#3.get the shared vocabulary
#python3 src/kg_info.py\
#    --swow_file /home/chunhua/Commonsense/SWOWEN-2018/data/2018/SWOW-EN.R100.csv\
#    --vocab_shared ./data/vocab_shared.txt\
#    --conceptnet_quadruple ./data/concept_quadruple\
#    --conceptnet_quadruple_filter ./data/concept_quadruple.filter\
#    --swow_triple ./data/swow_triple\
#    --swow_triple_filter ./data/swow_triple.filter
#    #--conceptnet_tuple_file /home/chunhua/Commonsense/comet-commonsense/data/conceptnet/train100k.txt
#
