import gc
import multiprocessing
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from openea.modules.finding.similarity import sim
from openea.modules.utils.util import task_divide, merge_dic
import openea.modules.load.read as rd


def calculate_rank_bidirection(gold_hrt, pred_tails, pred_heads, top_k, hr_to_multi_t, tr_to_multi_h, filter_rank=True, suffix='valid'):
    mr_12, mrr_12, hits_12, hits_12_list = calculate_rank(gold_hrt, pred_tails, top_k, hr_to_multi_t, 'hr->t', filter_rank)
    mr_21, mrr_21, hits_21, hits_21_list= calculate_rank(gold_hrt, pred_heads, top_k, tr_to_multi_h, 'h<-rt', filter_rank)

    mr = round((mr_12 + mr_21)/2, 4)
    mrr = round((mrr_12+ mrr_21)/2, 4)
    hits = ((hits_12 + hits_21)/2).tolist() 
    for i in range(len(hits)):
        hits[i] = round(hits[i], 4)
    hits=np.array(hits)
    if "test1" in suffix or "test2" in suffix:
        print("mr_12:{}, mrr_12:{}, hits_12:{}".format(mr_12, mrr_12, hits_12))
        print("mr_21:{}, mrr_21:{}, hits_21:{}".format(mr_21, mrr_21, hits_21))
    return mr, mrr, hits, hits_12_list, hits_21_list


def calculate_rank(gold_hrt, dis_mat, top_k, filtered, direction='hr->t', filter_rank=True):
    t = time.time()
    assert 1 in top_k
    hits_all = list()
    mr = list() 
    mrr = list() 
    hits10_rest = list()
    ranks_k = list() 

    large_number = 1e3

    for i, (h,r,t) in enumerate(gold_hrt):
        score = dis_mat[i,:]

        if direction=='hr->t':
            gold = t
            source = h
        elif direction=='h<-rt':
            gold = h 
            source= t
        
        #remove self-conncection
        if source!=gold:
            score[source] = large_number 

        #set large number for filtered ranking
        if filter_rank:
            for idx in filtered.get((source, r)):
                if idx!=gold:
                    score[idx] = large_number

        rank = score.argsort()

        hits = [0] * len(top_k)
        rank_index = np.where(rank == gold)[0][0]
        mr.append([rank_index + 1])
        mrr.append( [1 / (rank_index + 1)])

        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                hits[j] += 1
        hits_all.append(hits)

        rank_score = [np.round(score[j], 4) for j in rank[0:10]] 
        hits10_rest.append((gold, zip(rank[0:10], rank_score)))
    mr = np.mean(mr)
    mrr = np.mean(mrr)
    hits = np.mean(hits_all, axis=0)

    #cost = time.time() - t
    #print("accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.4f}, rank_candidates: {}, time = {:.3f} s. ".
    #                format(top_k, hits, mr, mrr, num, cost))
    del dis_mat
    return mr, mrr, hits, hits10_rest



def pairs_id2ent(inp_pairs, id1_entities, id2_entities):
    #out_pairs=set()
    out_pairs= list()
    for id1, id2s in inp_pairs:
        assert id1 in id1_entities.keys()
        ent1 = id1_entities[id1]
        ent2s=list()
        for id2, score in id2s:
            #id2 = id2[0] #(id,)
            #score = score[0] #(id,)
            #print(id2, score)
            assert id2 in id2_entities.keys()
            ent2s.append((id2_entities[id2], score))
        out_pairs.append((ent1, ent2s))
    return out_pairs

def write_rank_to_file(kg_eval, t_pred, h_pred, id_entities_dict, out_folder, suffix, pos_triples=None):

    if pos_triples is None:
        pos_triples = kg_eval.relation_triples_list 

    id2ent= kg_eval.id_entities_dict
    id2rel = kg_eval.id_relations_dict

    t_pred_ents = rd.pairs_id2ent(t_pred, id_entities_dict, id_entities_dict)
    h_pred_ents = rd.pairs_id2ent(h_pred, id_entities_dict, id_entities_dict)

    file_t = out_folder + suffix + "_t_prediction.csv"
    file_h = out_folder + suffix + "_h_prediction.csv"

    h_lines=list()
    t_lines=list()

    for i, x in enumerate(pos_triples):
        t_line= [(id2ent[x[0]], id2rel[x[1]], id2ent[x[2]])]
        h_line= [(id2ent[x[0]], id2rel[x[1]], id2ent[x[2]])]

        for x in t_pred_ents[i][1]:
            t_line.extend(x)

        for x in  h_pred_ents[i][1]:
            h_line.extend(x)

        t_lines.append(t_line)
        h_lines.append(h_line)
    write2csv(t_lines, file_t)
    write2csv(h_lines, file_h)
    print("save {}\nsave {}".format(file_t, file_h))


def write2csv(result, file_name):
    COLUMNS = ["gold_triples"]
    for i in range(1,11):
        COLUMNS.append("rank-{}".format(i))
        COLUMNS.append("score-{}".format(i))
    df=pd.DataFrame(result, columns=COLUMNS)
    df.to_csv(file_name, index=True)
