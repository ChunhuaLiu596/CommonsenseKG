#!/usr/bin/env bash

method="iptranse"
gpu=7
data="all"
split="all"
log_folder="../../output/log/"
mode='full'

while getopts "m:g:d:s:o:l:" opt;
do
    case ${opt} in
        m) method=$OPTARG ;;
        g) gpu=$OPTARG ;;
        d) data=$OPTARG ;;
        s) split=$OPTARG ;;
        o) mode=$OPTARG ;;
        l) log_folder=$OPTARG ;;
        *) echo "parameter error";;
    esac
done

args_folder="args/"
echo "log folder: " "${log_folder}"
training_data=("C_S_V0")

echo "training data: " "${training_data[@]}"
#data_splits=('721_5fold/1/' '721_5fold/2/' '721_5fold/3/' '721_5fold/4/' '721_5fold/5/')
data_splits=('271_5fold/1/')

if [[ ${split} == "1" ]]; then
    data_splits=('721_5fold/1/')
elif [[ ${split} == "2" ]]; then
    data_splits=('721_5fold/2/')
elif [[ ${split} == "3" ]]; then
    data_splits=('721_5fold/3/')
elif [[ ${split} == "4" ]]; then
    data_splits=('721_5fold/4/')
elif [[ ${split} == "5" ]]; then
    data_splits=('721_5fold/5/')
fi
echo "data splits: " "${data_splits[@]}"

py_code='main_from_args.py'
if [[ ${mode} = "wo_attr" ]]; then
    py_code='main_from_args_wo_attr.py'
elif [[ ${mode} = "test" ]]; then
    py_code='main_from_args_test.py'
    log_folder="../../output/test_log/"
elif [[ ${mode} = "rev" ]]; then
    py_code='main_from_args_reversed.py'
    log_folder="../../output/rev_log/"
fi
echo "py code: " "${py_code}"

#beta1s=(0.1 0.11 0.12 0.125 0.13 0.14 0.145 0.15 0.16 0.17 0.175 0.18 0.19 0.2)

beta1s=(2)
beta2s=(1 1 1 1)
lrs=(0.07 0.07 0.07 0.07)
train_kgs=("kg12" "kg2" "kg12" "kg12")
predict_relations=(true true true true)
align_modules=("sharing" "non-sharing" "non-sharing" "sharing")
maximum_count=3
count=0


for data_name in "${training_data[@]}"; do
    echo ""
    echo "${data_name}"
    log_folder_current=${log_folder}${method}/${data_name}/
    if [[ ! -d ${log_folder_current} ]];then
        mkdir -p "${log_folder_current}"
        echo "create log folder: " "${log_folder_current}"
    fi
    for data_split in "${data_splits[@]}"; do
        echo "${data_split}"
        args_file=${args_folder}${method}"_args_15K_rel.json"
        log_div=${data_split//\//_}

        for ((idx=0; idx<${#beta1s[@]}; idx++));do
            beta1=${beta1s[$idx]}
            beta2=${beta2s[$idx]}
            lr=${lrs[$idx]}
            train_kg=${train_kgs[$idx]}
            predict_relation=${predict_relations[$idx]}
            align_module=${align_modules[$idx]}

            if [[ "$count" -eq "$maximum_count" ]]; then
                count=0
                echo "waiting for current cmds to finish ..."
                wait
            fi
            let count++
            cur_time="$(date +%Y%m%d%H%M%S)"
            echo "cur_time: ${cur_time}, beta1:${beta1}, beta2:${beta2}, lr:${lr}  count:${count}"
            CUDA_VISIBLE_DEVICES=${gpu} python3 ${py_code} "${args_file}" "${data_name}" "${data_split}" ${beta1} ${beta2} ${lr} "${train_kg}" ${predict_relation} "${align_module}" > "${log_folder_current}""${method}"_"${data_name}"_"${log_div}""${cur_time}" &
            sleep 1m
        done
    done
done



#grid_search
#for data_name in "${training_data[@]}"; do
#    echo ""
#    echo "${data_name}"
#    log_folder_current=${log_folder}${method}/${data_name}/
#    if [[ ! -d ${log_folder_current} ]];then
#        mkdir -p "${log_folder_current}"
#        echo "create log folder: " "${log_folder_current}"
#    fi
#    for data_split in "${data_splits[@]}"; do
#        echo "${data_split}"
#        args_file=${args_folder}${method}"_args_15K.json"
#        log_div=${data_split//\//_}
#        for beta1 in ${beta1s[@]};do
#            for beta2 in ${beta2s[@]}; do
#                if [[ "$count" -eq "$maximum_count" ]]; then
#                    count=0
#                    echo "waiting for current cmds to finish ..."
#                    wait
#                fi
#                let count++
#                cur_time="$(date +%Y%m%d%H%M%S)"
#                echo "cur_time: ${cur_time}, beta1:${beta1}, beta2:${beta2}, count:${count}"
#                CUDA_VISIBLE_DEVICES=${gpu} python3 ${py_code} "${args_file}" "${data_name}" "${data_split}" ${beta1} ${beta2} > "${log_folder_current}""${method}"_"${data_name}"_"${log_div}""${cur_time}" &
#                sleep 1m
#            done
#        done
#    done
#done
