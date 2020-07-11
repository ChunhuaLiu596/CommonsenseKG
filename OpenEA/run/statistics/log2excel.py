from xlwt import Workbook
import os,sys
import numpy as np
import ast
import re
import json
import pandas as pd

# FOLDER_PREFIX = '/media/sl/Data/workspace/VLDB2020/'
#FOLDER_PREFIX = '/Users/sloriac/code/VLDB/'
FOLDER_PREFIX = '/home/chunhua/Commonsense/OpenEntitiyAlignment/'
COLUMNS=['filename', 'train_kg', 'predict_relation', 'alignment_module', 'beta1', 'beta2', 'learning_rate','margin',
        'rel_inp_dropout', 'rel_hid_dropout', 'rel_hidden_dim', 'tensorboard', 
        'a-hits1','a-hits5','a-hits10','a-hits50','a-mr', 'a-mrr',
        'c-hits1','c-hits5','c-hits10','c-hits50','c-mr', 'c-mrr', 
        'rel-acc']
COLUMN_NUM = len(COLUMNS) 

def check_folder_path(path_str):
    assert path_str.startswith('results output folder:')
    print("results output folder:", path_str)
    folder_path = FOLDER_PREFIX + path_str.strip('\n').split('../../')[-1]
    if not os.path.exists(folder_path):
        print(folder_path)
    if not os.path.exists(folder_path):
        print("{} doesn't exist".format(folder_path))

    #assert os.path.exists(folder_path)
    folder_path = folder_path.split('/2019')[0] + '/'
    folder_cnt = 0
    print("folder_path", folder_path)
    for _ in os.listdir(folder_path):
        if _ != ".DS_Store":
            folder_cnt = folder_cnt + 1
    #if folder_cnt != 1:
    #    print(folder_path)
    #    print('Wrong:', folder_cnt, 'folders exist!')
    #assert folder_cnt == 1


def judge(str1, str2):
    str1 = str1.strip('\n').split(', ')
    str2 = str2.strip('\n').split(', ')
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            print(str1[i], '\t\t', str2[i])


def write2excel(result, book, sheet_name):
    sheet = book.add_sheet(sheet_name)
    sheet.write(1, 1, 'Filename')
    sheet.write(1, 2, 'beta1')
    sheet.write(1, 3, 'beta2')
    sheet.write(1, 4, 'Alignment')
    sheet.write(1, 5, 'Alignment')
    sheet.write(1, 6, 'Alignment')
    sheet.write(1, 7, 'Alignment')
    sheet.write(1, 8, 'Alignment')
    sheet.write(1, 9, 'Alignment')
    sheet.write(1, 10, 'Completion')
    sheet.write(1, 11, 'Completion')
    sheet.write(1, 12, 'Completion')
    sheet.write(1, 13, 'Completion')
    sheet.write(1, 14, 'Completion')
    sheet.write(1, 15, 'Completion')
    #sheet.write(1, 16, 'REL_ACC')
    sheet.write(2, 1, 'Filename')
    sheet.write(2, 2, 'beta1')
    sheet.write(2, 3, 'beta2')
    sheet.write(2, 4, 'Hits1')
    sheet.write(2, 5, 'Hits5')
    sheet.write(2, 6, 'Hits10')
    sheet.write(2, 7, 'Hits50')
    sheet.write(2, 8, 'MR')
    sheet.write(2, 9, 'MRR')
    sheet.write(2, 10, 'Hits1')
    sheet.write(2, 11, 'Hits5')
    sheet.write(2, 12, 'Hits10')
    sheet.write(2, 13, 'Hits50')
    sheet.write(2, 14, 'MR')
    sheet.write(2, 15, 'MRR')
    #sheet.write(2, 16, 'REL_ACC')
    print("Total {} files".format(len(result)))
    for i in range(len(result)):
        for j in range(COLUMN_NUM):
            sheet.write(i+3, j+2, result[i][j])

def write2csv(result, file_name):
    df=pd.DataFrame(result, columns=COLUMNS)
    df.to_csv(file_name, index=False)

def extract_results(file_path, is_tune_param=False, is_csls=False):
    def extract_results_from_line(res_line):
        res = []
        res_line = res_line.strip('\n').split(' = ')
        #hits = res_line[1].lstrip('[').split(']')[0].split(' ')
        hits = res_line[1].lstrip('[').split(']')[0]
        hits = re.split(' |, ', hits) 
        for hit in hits:
            if hit != '':
                #res.append(float(hit)/100)
                #res.append((float(hit)/100)*100)
                res.append(float(hit))
        res.append(float(res_line[2].split(',')[0]))
        res.append(float(res_line[3].split(',')[0]))
        #print(len(res), res)
        return res

    def extract_rel_acc_results_from_line(res_line):
        #'#Results: test_rel_acc: 0.0162, triples: 1236 , cost time: 0.0075s'
        rel_acc = re.split(":|,| =", res_line.strip('\n'))[2]
        #print("rel_acc:{}".format(rel_acc))
        return [float(rel_acc)]

    def extract_args_results_from_line(res_line):
        res=[]
        args = re.split('(load arguments: )', res_line.strip('\n'))[2]
        args_dict = ast.literal_eval(str(args))

        res.append(args_dict.get('train_kg'))
        res.append(args_dict.get('predict_relation'))
        res.append(args_dict.get('alignment_module'))

        assert 'beta1' in args_dict
        assert 'beta2' in args_dict
        beta1 = args_dict.get('beta1')
        beta2 = args_dict.get('beta2')
        res.append(beta1)
        res.append(beta2)

        res.append(args_dict.get('learning_rate'))
        res.append(args_dict.get('margin'))
        res.append(args_dict.get('rel_inp_dropout'))
        res.append(args_dict.get('rel_hid_dropout'))
        res.append(args_dict.get('rel_hidden_dim'))

        #print("beta1: {}, beta2: {}".format(beta1, beta2))
        return res

    def extract_tensorboard_from_line(res_line):
        '''Open tensorboard: tensorboard --logdir=run1:"../../output/results/IPTransE/C_S_V6/271_5fold/1/20200615212403/train/",run2:"../../output/results/IPTransE/C_S_V6/271_5fold/1/20200615212403/valid/",run3:"../../output/results/IPTransE/C_S_V6/271_5fold/1/20200615212403/test/" '''
        res=[]
        tensorboard = re.split('(Open tensorboard: )', res_line.strip('\n'))[2]
        res.append(tensorboard)
        #print("beta1: {}, beta2: {}".format(beta1, beta2))
        return res

    prefix_str_align = 'alignment results: '
    prefix_str_com = 'completion results: '
    prefix_str_rel = 'relation results: '
    if is_csls:
        prefix_str = 'accurate results with csls: '
    results =[]
    is_final_result=False
    # results.extend([['Filename','beta1','beta2','align-hits2', 'align-hits5', 'align-hits10', 'align-hits50', 'align-mr','align-mrr', 
    #                    'comp-hits1',  'comp-hits5', 'comp-hits10', 'comp-hits50', 'comp-mr', 'comp-mrr', 'rel-acc']])
    with open(file_path, 'r', encoding='utf-8') as file:
        # is_final_result = False
        # res_line_current = ''
        first_output_folder_line = True
        completion_printed = False
        for line in file:
            #if line.startswith('Training ends.') or 'should early stop' in line:
            if line.startswith('Start testing'):
                is_final_result = True
            #elif line.startswith('epoch 990,'):
                
                #if is_tune_param:
                #    return extract_results_from_line(res_line_current)
            #     continue

            if first_output_folder_line and line.startswith('. output folder'):
                check_folder_path(line)
                first_output_folder_line = False

            if first_output_folder_line and line.startswith('load arguments:'):
                results.extend(extract_args_results_from_line(line))

            if first_output_folder_line and line.startswith('Open tensorboard:'):
                results.extend(extract_tensorboard_from_line(line))

            if line.startswith(prefix_str_align) and is_final_result:
                res_line_current = line
                results.extend(extract_results_from_line(line))

            if line.startswith(prefix_str_com) and is_final_result and not completion_printed:
                res_line_current = line
                results.extend(extract_results_from_line(line))

            if first_output_folder_line and line.startswith(prefix_str_rel) and is_final_result and not completion_printed:
                results.extend(extract_rel_acc_results_from_line(line))
                completion_printed=True

    return results


def run(folder, model_name, is_csls=False, dataset_type='15K'):
    assert os.path.exists(folder), "{}".format(folder)
    book = Workbook(encoding='utf-8')

    files = set()
    #exclude_dirs=set(["#"])
    #for root, dirs, files in os.walk(folder+'/', topdown=True):
    #    dirs[:] = [d for d in dirs if d not in exclude_dirs]
#    for file_folder in list(os.walk(folder+'/'))[1:]:
    for file_folder in list(os.walk(folder+dataset_type+'/'))[:]:
        #print ("file_folder: {} ".format(file_folder))
        if "#" in file_folder[2]:
            continue
        for file in file_folder[2]:
            if dataset_type in file:
                # files.add(file_folder[0]+file)
                files.add(file_folder[0]+'/'+file)

    files = sorted(files)

    result = []
    count =0
    result_incomplete = []
    for file in files:
        print("file: {}".format(file))
        res=[]
        row_name = [file.split('/'+model_name+'_')[-1]]
        res.extend(row_name)
        res_line = extract_results(file, is_csls=is_csls)
        print(res_line)
        res.extend(res_line)
    
        if len(res) !=COLUMN_NUM:
            if len(res) > 10:
                #if len(res) ==22:
                count+=1
                result_incomplete.append(res)
            continue
            #assert len(res)==10, "len_res:{}, res:{}".format(len(res),res)
        else:
            result.append(res)

    file_name = folder + model_name + '_' + dataset_type
    write2csv(result, file_name+'.csv')
    print("Write {} results to {}".format(len(result), file_name+'.csv'))
    if count>0:
        print("{} files don't satisfy".format(count))
        print("result_incomplete: {}".format(result_incomplete))
    #write2excel(result, book, model_name)
    #if is_csls:
    #    file_name += '_csls'
    #book.save(file_name+'.xlsx')
    #print("Write results to {} \nand {}".format(file_name+'.xlsx', file_name+'.csv'))
    return result

def compute_mean_variance(result, res_file='res.csv'):
    '''
    example: results=[['C_S_271_5fold_1_20200517100036', 0.00477, 0.01611, 0.02267, 0.06802, 754.387, 0.013748], ['C_S_271_5fold_2_20200517100658', 0.00596, 0.01431, 0.023849999999999996, 0.07156, 746.475, 0.015283], ['C_S_271_5fold_3_20200517101446', 0.01014, 0.02445, 0.03757, 0.08109999999999999, 751.717, 0.02109], ['C_S_271_5fold_4_20200517102134', 0.00895, 0.02088, 0.02804, 0.06683, 768.871, 0.018267], ['C_S_271_5fold_5_20200517111024', 0.00418, 0.01611, 0.02327, 0.06265, 765.524, 0.013117]]
    '''
    #print(result)
    result = np.split(np.array(result), 6, axis=1)[:]
    #result = ast.literal_eval(result)
    #result = np.array(result, dtype=float)
    mean = [ np.mean(x, dtype=np.float32) for x in result]
    std = [ np.std(x, dtype=np.float32) for x in result]
    assert len(std)==len(mean)
    
    metric_name=["Hits@1 (%)", "Hits@5 (%)", "Hits@10 (%)" ,"Hits@50 (%)", "MR", "MRR"]
    out=[]
    for i in range(len(mean)):
        print("{} avg and std: {:.4f}\t{:.4f}".format(metric_name[i],mean[i], std[i]))
        out.append("{:.4f}".format(mean[i]))
        out.append("{:.4f}".format(std[i]))
    out_line= "\t".join(out)
    return out_line


if __name__ == '__main__':
    '''
    method = 'mtranse'
    data_size = 'C_S_271'
    or    data_size = '15K'
    '''
    method=sys.argv[1]
    data_size=sys.argv[2]
    print(method, data_size)
    result1 = run('../../../output/log/'+method+'/', method, is_csls=False, dataset_type=data_size)
    #result2 = run('../../../output/log/'+method+'/', method, is_csls=True, dataset_type=data_size)

    #print("accurate results: ")
    #out1 = compute_mean_variance(result1,"res.csv")
    #print("\naccurate results with csls: ")
    #out2 = compute_mean_variance(result2, "res_csls.csv")

    ##os.rm
    #with open("res.csv", "w") as f:
    #    f.write(out1)
    #    f.write("\n")
    #    f.write(out2)