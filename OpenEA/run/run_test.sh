#!/usr/bin/env bash

#bash run_15K.sh -d ygv1 -o test -l "../../output/test_log/" -m attre
#bash run_15K.sh -d ygv1 -o test -l "../../output/test_log/" -m bootea
#bash run_15K.sh -d ygv1 -o test -l "../../output/test_log/" -m conve
#bash run_15K.sh -d ygv1 -o test -l "../../output/test_log/" -m gcnalign
#bash run_15K.sh -d ygv1 -o test -l "../../output/test_log/" -m imuse
#bash run_15K.sh -d ygv1 -o test -l "../../output/test_log/" -m iptranse
#bash run_15K.sh -d ygv1 -o test -l "../../output/test_log/" -m jape
#bash run_15K.sh -d ygv1 -o test -l "../../output/test_log/" -m kdcoe
#bash run_15K.sh -d ygv1 -o test -l "../../output/test_log/" -m mtranse
#bash run_15K.sh -d ygv1 -o test -l "../../output/test_log/" -m multike
#bash run_15K.sh -d ygv1 -o test -l "../../output/test_log/" -m rdgcn
#bash run_15K.sh -d ygv1 -o test -l "../../output/test_log/" -m rsn4ea
#bash run_15K.sh -d ygv1 -o test -l "../../output/test_log/" -m rotate
#bash run_15K.sh -d ygv1 -o test -l "../../output/test_log/" -m sea
#bash run_15K.sh -d ygv1 -o test -l "../../output/test_log/" -m transh


#bash run_15K.sh -d C_S_V6 -o test -l "../../output/test_log/" -m iptranse -g 6
#bash run_15K.sh -d C_S_V6 -o test -l "../../output/test_log/" -m iptranse -g 6
#CUDA_VISIBLE_DEVICES=7 python main_from_args_test.py args/iptranse_args_15K_rel.json C_S_V6 271_5fold/1/20200624141801/ 0 0 0.08
#CUDA_VISIBLE_DEVICES=7 python main_from_args_test.py args/iptranse_args_15K_rel.json C_S_V6 271_5fold/1/20200627042700/ 0 0 0.08
#CUDA_VISIBLE_DEVICES=6 python3 main_from_args_test.py args/iptranse_args_15K_test.json C_S_V0 271_5fold/1/20200705212210/ 0 0 0.08
CUDA_VISIBLE_DEVICES=6 python main_from_args_test.py args/iptranse_args_15K_test.json C_S_V0 271_5fold/1/20200708125002/ 0 0 0.08

