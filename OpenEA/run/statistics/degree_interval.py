from utils import *
import os
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter

sns.set(style="darkgrid")
save_name = "log/histogram.png"

def create_entity_dicts(all_tuples):
    e1_to_multi_e2 = {}
    e2_to_multi_e1 = {}
    e12_to_multi_r={}
    ent_set=set()
    ent_to_freq={}

    for tup in all_tuples:
        e1, rel, e2 = tup
        ent_set.add(e1)
        ent_set.add(e2)
        if e1 not in ent_to_freq:
            ent_to_freq[e1] = 1
        else:
            ent_to_freq[e1] +=1

        if e2 not in ent_to_freq:
            ent_to_freq[e2] =1
        else:
            ent_to_freq[e2] +=1

        if (e1, rel) in e1_to_multi_e2:
            e1_to_multi_e2[(e1, rel)].append(e2)
        else:
            e1_to_multi_e2[(e1, rel)] = [e2]

        if (e2, rel) in e2_to_multi_e1:
            e2_to_multi_e1[(e2, rel)].append(e1)
        else:
            e2_to_multi_e1[(e2, rel)] = [e1]

        if (e1, e2) in e12_to_multi_r:
            e12_to_multi_r[(e1, e2)].append(rel)
        else:
            e12_to_multi_r[(e1, e2)]= [rel]

    print("triple_num: {}, entity_num: {}".format(len(all_tuples), len(ent_set)))

    count_1_1=0
    count_1_n=0
    count_n_1=0
    count_n_n=0

    for k, e2s in e1_to_multi_e2.items():
        if len(e2s)==1:
            count_1_1 +=1
        if len(e2s)>1:
            #print(k, e2s)
            count_1_n +=1
    print("e1_to_multi_e2. count_1_1: {}, count_1_n: {}".format(count_1_1, count_1_n))

    count_1_1=0
    for k, e1s in e2_to_multi_e1.items():
        if len(e1s)==1:
            count_1_1 +=1
        if len(e1s)>1:
            #print(k, e2s)
            count_n_1 +=1
    print("e2_to_multi_e1. count_1_1: {}, count_n_1: {}".format(count_1_1, count_n_1))

    count_1_1=0
    for e12, rs in e12_to_multi_r.items():
        if len(rs)==1:
            count_1_1 +=1
        if len(rs)>1:
            print(e12, rs)
            count_n_n +=1
    print("e12_to_multi_r. count_1_1: {}, count_n_n: {}".format(count_1_1, count_n_n))
    print('-------')

    #print("e1_to_multi_e2: {}".format( e1_to_multi_e2))
    #print("e2_to_multi_e1: {}".format( e2_to_multi_e1))
    return e1_to_multi_e2, e2_to_multi_e1, ent_to_freq, ent_set


def count_ent_degree(triples, is_sorted=False):
    ent_degree = {}
    for (h, _, t) in triples:
        degree = 1
        if h in ent_degree:
            degree += ent_degree[h]
        ent_degree[h] = degree

        degree = 1
        if t in ent_degree:
            degree += ent_degree[t]
        ent_degree[t] = degree

    if is_sorted:
        ent_degree = sorted(ent_degree.items(), key=lambda d: d[1], reverse=True)
        return {e:c for (e, c) in ent_degree}
    return ent_degree


def filter_pairs_by_degree_interval(pair_degree, degree_interval):
    pair_set = set()
    for pair, degree in pair_degree.items():
        if degree_interval[0] <= degree < degree_interval[1]:
            pair_set.add(pair)
    return pair_set


def gold_standard_compare(gold_set, exp_set):

    right_set = gold_set & exp_set
    print(len(right_set), len(exp_set), len(gold_set))
    if len(right_set) == 0:
        return 0, 0, 0
    p = len(right_set) / len(exp_set)
    r = len(right_set) / len(gold_set)
    f1 = 2*p*r / (p+r)
    return p, r, f1


def count_pair_degree(ent_degree_1, ent_degree_2, links):
    pair_degree = {}
    for (e1, e2) in links:
        pair_degree[(e1, e2)] = (ent_degree_1[e1] + ent_degree_2[e2]) / 2
    return pair_degree

def get_ent_freq(ent_to_freq, ent_set, freq_threshold=250):
    subset_ent_to_freq = Counter()
    for ent in ent_set:
        if ent in ent_to_freq:
            freq = ent_to_freq[ent]
            if freq>=250:
                freq=250
            subset_ent_to_freq[ent] = freq
    #print("subset_ent_to_freq: {}".format(subset_ent_to_freq))
    return subset_ent_to_freq

def get_freq_distribution(data):
    cnt = Counter()
    for (k,v) in data.items():
        cnt[v] += 1
    return cnt

def calcualte_data_freq(dataset):
    data_folder = '../../../datasets/'+dataset+'/'
    print("dataset: {}".format(data_folder))
    rel_triples_1, _, _ = read_relation_triples(data_folder + 'rel_triples_1')
    rel_triples_2, _, _ = read_relation_triples(data_folder + 'rel_triples_2')
    rel_triples_overlap12, _, _ = read_relation_triples(data_folder + 'rel_triples_overlap12')

    rel_triples_valid, _, _ = read_relation_triples(data_folder + 'rel_triples_valid')
    rel_triples_test, _, _ = read_relation_triples(data_folder + 'rel_triples_test')

    _, _, ent_to_freq1, ent_set1 = create_entity_dicts(rel_triples_1|rel_triples_valid|rel_triples_test)
    _, _, ent_to_freq2, ent_set2 = create_entity_dicts(rel_triples_2|rel_triples_valid|rel_triples_test)
    _, _, ent_to_freq12, ent_set12 = create_entity_dicts(rel_triples_overlap12)

    _, _, ent_to_freq_valid, ent_set_valid = create_entity_dicts(rel_triples_valid)
    _, _, ent_to_freq_test, ent_set_test = create_entity_dicts(rel_triples_test)

    #freq on whole kg1
    #ent_set_valid_freq = get_ent_freq(ent_to_freq1, ent_set_valid)
    #ent_set_test_freq = get_ent_freq(ent_to_freq1, ent_set_test)
    def _get_freq_distribution(ent_to_freq_all):
        ent_set_valid_freq = get_ent_freq(ent_to_freq_all, ent_set_valid)
        ent_set_test_freq = get_ent_freq(ent_to_freq_all, ent_set_test)
        cnt_valid = get_freq_distribution(ent_set_valid_freq)
        cnt_test = get_freq_distribution(ent_set_test_freq)
        return cnt_valid, cnt_test

    cnt_valid12, cnt_test12 = _get_freq_distribution(ent_to_freq12)
    cnt_valid1, cnt_test1 = _get_freq_distribution(ent_to_freq1)
    cnt_valid2, cnt_test2 = _get_freq_distribution(ent_to_freq2)

    #def _plot_freq_distribution(ent_to_freq_all , title, save_name):
    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(64, 32))
    all_keys = set(cnt_test12.keys())|set(cnt_valid12.keys())| set(cnt_test1.keys())|set(cnt_valid1.keys())|set(cnt_test2.keys())|set(cnt_valid2.keys())

    def _merge_keys(keys, dic):
        for k in keys:
            if k not in dic:
                dic[k]=0
        return dic

    cnt_valid12 = _merge_keys(all_keys, cnt_valid12)
    cnt_valid1 =  _merge_keys(all_keys, cnt_valid1)
    cnt_valid2 = _merge_keys(all_keys, cnt_valid2)
    cnt_test12 = _merge_keys(all_keys, cnt_test12)
    cnt_test1 =  _merge_keys(all_keys, cnt_test1)
    cnt_test2 = _merge_keys(all_keys, cnt_test2)

    d = {"entity_freq": list(all_keys), 
        "t-count-12": list(cnt_test12.values()),
        "v-count-12": list(cnt_valid12.values()),
        "t-count-1": list(cnt_test1.values()),
        "v-count-1": list(cnt_valid1.values()),
        "t-count-2": list(cnt_test2.values()),
        "v-count-2": list(cnt_valid2.values()),
    }
    df = pd.DataFrame(data=d)

    ax1 = sns.barplot(x="entity_freq", y="v-count-12", data=df, ax=ax1)
    ax1.set_xlabel("Valid_Entity_Freq_on_OverlapKG" )
    ax1.set_ylabel("Entity-Num")
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax2 = sns.barplot(x="entity_freq", y="t-count-12", data=df, ax=ax2)
    ax2.set_xlabel("Test_Entity_Freq_on_OverlapKG" )
    ax2.set_ylabel("Entity-Num" )
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    #ax2.set_title(title)
    #ax2 = sns.relplot(x="entity_freq", y="t-count", kind="line", data=df)
    ax3 = sns.barplot(x="entity_freq", y="v-count-1", data=df, ax=ax3)
    ax3.set_xlabel("Valid_Entity_Freq_on_ConceptNet" )
    ax3.set_ylabel("Entity-Num-on-ConceptNet")
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax4 = sns.barplot(x="entity_freq", y="t-count-1", data=df, ax=ax4)
    ax4.set_xlabel("Test_Entity_Freq_on_ConceptNet" )
    ax4.set_ylabel("Entity-Num" )
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax4.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax5 = sns.barplot(x="entity_freq", y="v-count-2", data=df, ax=ax5)
    ax5.set_xlabel("Valid_Entity_Freq_on_SWOW" )
    ax5.set_ylabel("Entity-Num")
    ax5.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax5.xaxis.set_major_formatter(ticker.ScalarFormatter())

    ax6 = sns.barplot(x="entity_freq", y="t-count-2", data=df, ax=ax6)
    ax6.set_xlabel("Test_Entity_Freq_on_SWOW" )
    ax6.set_ylabel("Entity-Num" )
    ax6.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax6.xaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.tight_layout() 
    save_name = 'log/{}_valid_test_ent_freq_distributions.png'.format(dataset)
    plt.savefig(save_name, format='png')
    print("save {}".format(save_name))


def calcualte_data_degree(dataset):
    #ent_set_valid_freq = get_ent_freq(ent_to_freq2, ent_set_valid)
    #ent_set_test_freq = get_ent_freq(ent_to_freq2, ent_set_test)

    #ent_degree_1 = count_ent_degree(rel_triples_1, is_sorted=True)
    #ent_degree_2 = count_ent_degree(rel_triples_2, is_sorted=True)
    #ent_degree_valid = count_ent_degree(rel_triples_valid, is_sorted=True)
    #ent_degree_test = count_ent_degree(rel_triples_test, is_sorted=True)
    #cnt_valid = get_freq_distribution(ent_degree_valid)
    #cnt_test = get_freq_distribution(ent_degree_test)
    #average_count = sum(np.array(list(cnt.keys()))*np.array(list(cnt.values())))/ sum(np.array(list(cnt.values())))
    #rint("average_degree: {}. cnt: {}".format( average_degree, cnt))

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    keys = set(cnt_test.keys())|set(cnt_valid.keys())
    for k in keys:
        if k not in cnt_test:
            cnt_test[k]=0
        if k not in cnt_valid:
            cnt_valid[k]=0

    d = {"test_entity_degree":list(cnt_test.keys()), "t-count": list(cnt_test.values()),
        "valid_entity_degree":list(cnt_valid.keys()), "v-count": list(cnt_valid.values())}
    df = pd.DataFrame(data=d)
    ax2 = sns.barplot(x="test_entity_degree", y="t-count", data=df, ax=ax2)
    ax2.set_ylabel("Entity-Num")
    
    df = pd.DataFrame(data=d)
    ax1 = sns.barplot(x="valid_entity_degree", y="v-count", data=df, ax=ax1)
    ax1.set_ylabel("Entity-Num")
    plt.savefig('log/{}_valid_test_ent_degree_distribution.png'.format(dataset), format='png')

    #ent_links_valid = read_links(data_folder+'ent_links_valid')
    #pair_degree_gold = count_pair_degree(ent_degree_1, ent_degree_2, ent_links_valid)
    #print("pari_degree_valid:{}".format(pair_degree_gold))

    #ent_links = read_links(data_folder+'/'+'ent_links_test')
    #pair_degree_gold = count_pair_degree(ent_degree_1, ent_degree_2, ent_links)
    #print("pari_degree_test:{}".format(pair_degree_gold))


def run(dataset, data_split, method, degree_interval=None):
    if degree_interval is None:
        degree_interval = [1, 6, 11, 16, 21, 1000000]
    data_folder = '../../../datasets/'+dataset+'/'
    result_folder = '../../../output/results/'+method+'/'+dataset+'/'+data_split+'/'
    result_folder += list(os.walk(result_folder))[0][1][0] + '/'
    assert os.path.exists(result_folder)
    assert os.path.exists(data_folder)

    rel_triples_1, _, _ = read_relation_triples(data_folder + 'rel_triples_1')
    rel_triples_2, _, _ = read_relation_triples(data_folder + 'rel_triples_2')
    rel_triples_valid, _, _ = read_relation_triples(data_folder + 'rel_triples_valid')
    rel_triples_test, _, _ = read_relation_triples(data_folder + 'rel_triples_test')

    ent_degree_1 = count_ent_degree(rel_triples_1)
    ent_degree_2 = count_ent_degree(rel_triples_2)
    ent_degree_valid = count_ent_degree(rel_triples_valid)
    ent_degree_test = count_ent_degree(rel_triples_test)

    ent_links = read_links(data_folder+'/'+data_split+'/'+'ent_links_test')
    pair_degree_gold = count_pair_degree(ent_degree_1, ent_degree_2, ent_links)

    ent_links_valid = read_links(data_folder+'/'+data_split+'/'+'ent_links_valid')
    pair_degree_gold = count_pair_degree(ent_degree_1, ent_degree_2, ent_links_valid)


    #id_ent_dict_1, id_ent_dict_2 = id2ent_by_ent_links_index(ent_links)
    #aligned_ent_id_pair_set = read_alignment_results(result_folder+'alignment_results_12')
    #aligned_ent_pair_set = set([(id_ent_dict_1[e1], id_ent_dict_2[e2]) for (e1, e2) in aligned_ent_id_pair_set])
    #pair_degree_exp = count_pair_degree(ent_degree_1, ent_degree_2, aligned_ent_pair_set)

    #pairs_gold = filter_pairs_by_degree_interval(pair_degree_gold, [1, 1000000])
    #pairs_exp = filter_pairs_by_degree_interval(pair_degree_exp, [1, 1000000])
    #p, r, f1 = gold_standard_compare(pairs_gold, pairs_exp)
    #print('[%d, %d): [P, R, F1] = [%.4f, %.4f, %.4f]' % (1, 1000000, p, r, f1))

    #f1s = []
    #ps = []
    #rs = []
    #for i in range(len(degree_interval)-1):
    #    pairs_gold = filter_pairs_by_degree_interval(pair_degree_gold, [degree_interval[i], degree_interval[i+1]])
    #    pairs_exp = filter_pairs_by_degree_interval(pair_degree_exp, [degree_interval[i], degree_interval[i+1]])
    #    p, r, f1 = gold_standard_compare(pairs_gold, pairs_exp)
    #    print('[%d, %d): [P, R, F1] = [%.4f, %.4f, %.4f]' % (degree_interval[i], degree_interval[i+1], p, r, f1))
    #    f1s.append(f1)
    #    ps.append(p)
    #    rs.append(r)
    #return ps, rs, f1s


if __name__ == '__main__':
    #dataset = 'DBP_en_DBP_fr_15K_V1'
    datasets = ['C_S_V0']
    for dataset in datasets:
        #calcualte_data_degree(dataset)
        calcualte_data_freq(dataset)
    #data_split = '721_5fold/1'
    #p_r_f1 = 'r'
    #methods = ['MTransE', 'IPTransE', 'JAPE', 'KDCoE', 'BootEA', 'GCN_Align', 'AttrE', 'IMUSE', 'SEA', 'RSN4EA',
    #           'MultiKE', 'RDGCN']
    #res = [[0 for i in range(len(methods))] for j in range(4)]
    #cnt = 0
    #for method in methods:
    #    ps, rs, f1s = run(dataset, data_split, method, degree_interval=[1, 6, 11, 16, 1000000])
    #    results = ps
    #    if p_r_f1 == 'r':
    #        results = rs
    #    elif p_r_f1 == 'f1':
    #        results = f1s
    #    res[0][cnt] = results[0]
    #    res[1][cnt] = results[1]
    #    res[2][cnt] = results[2]
    #    res[3][cnt] = results[3]
    #    cnt += 1
    #for i in range(4):
    #    output = ''
    #    for j in range(len(methods)):
    #        output += str(res[i][j])
    #        if j != len(methods) - 1:
    #            output += '\t'
    #    print(output)

