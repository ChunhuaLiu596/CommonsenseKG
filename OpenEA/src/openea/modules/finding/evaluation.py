import numpy as np

from openea.modules.finding.alignment import greedy_alignment
import time

def rank_alignment(embeds1, embeds2, mapping, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=True):
    t = time.time()
    if mapping is None:
        mr_12, mrr_12, hits_12, hits_12_list = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                                        metric, normalize, csls_k, accurate)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        mr_12, mrr_12, hits_12, hits_12_list = greedy_alignment(test_embeds1_mapped, embeds2, top_k, threads_num,
                                                                        metric, normalize, csls_k, accurate)

    cost = time.time() - t
    rank_candidates_num = len(hits_12_list)
    if accurate:
        print("alignment results: hits@{} = {}, mr = {:.4f}, mrr = {:.4f}, rank_candidates: {}, time = {:.3f} s. ".
                    format(top_k, hits_12, mr_12, mrr_12, rank_candidates_num, cost))
    return mr_12, mrr_12, hits_12, hits_12_list


def rank_alignment_bidirection(embeds1, embeds2, mapping, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=True):
    t = time.time()
    if mapping is None:
        mr_12, mrr_12, hits_12, hits_12_list = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                        metric, normalize, csls_k, accurate)

        mr_21, mrr_21, hits_21, hits_21_list = greedy_alignment(embeds2, embeds1, top_k, threads_num,
                                                        metric, normalize, csls_k, accurate)
    else:                                                 
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        mr_12, mrr_12, hits_12, hits_12_list = greedy_alignment(test_embeds1_mapped, embeds2, top_k, threads_num,
                                                                metric, normalize, csls_k, accurate)

        mr_21, mrr_21, hits_21, hits_21_list = greedy_alignment(embeds2, test_embeds1_mapped, top_k, threads_num,
                                                        metric, normalize, csls_k, accurate)

    mr = round((mr_12 + mr_12)/2, 4)
    mrr = round((mrr_12 + mrr_21)/2, 4)
    hits = ((hits_12 + hits_21)/2).tolist() 
    for i in range(len(hits)):
        hits[i] = round(hits[i], 4)
    hits=np.array(hits)

    cost = time.time() - t
    rank_candidates_num = len(hits_12_list)
    if accurate:
        if csls_k > 0:
            print("alignment results with csls: csls={}, hits@{} = {}, mr = {:.4f}, mrr = {:.4f}, rank_candidates: {}, time = {:.3f} s ".
                    format(csls_k, top_k, hits, mr, mrr, rank_candidates_num, cost))
        else:
            print("alignment results: hits@{} = {}, mr = {:.4f}, mrr = {:.4f}, rank_candidates: {}, time = {:.3f} s. ".
                    format(top_k, hits, mr, mrr, rank_candidates_num, cost))
    else:
        if csls_k > 0:
            print("quick results with csls: csls={}, hits@{} = {}, time = {:.4f} s ".format(csls_k, top_k, hits, cost))
        else:
            print("alignment results: hits@{} = {}, mr = {:.4f}, mrr = {:.4f}, rank_candidates: {}, time = {:.3f} s. ".
                    format(top_k, hits, mr, mrr, rank_candidates_num, cost))
    return  mr, mrr, hits, hits_12_list, hits_21_list



def early_stop(flag1, flag2, flag):
    if flag <= flag2 <= flag1:
        print("\n == should early stop == \n")
        return flag2, flag, True
    else:
        return flag2, flag, False



def valid(embeds1, embeds2, mapping, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=False):
    if mapping is None:
        _, hits1_12, mr_12, mrr_12 = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                      metric, normalize, csls_k, accurate)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        _, hits1_12, mr_12, mrr_12 = greedy_alignment(test_embeds1_mapped, embeds2, top_k, threads_num,
                                                      metric, normalize, csls_k, accurate)
    return hits1_12, mrr_12


def test(embeds1, embeds2, mapping, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=True):
    if mapping is None:
        alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                                      metric, normalize, csls_k, accurate)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(test_embeds1_mapped, embeds2, top_k, threads_num,
                                                                      metric, normalize, csls_k, accurate)
    return alignment_rest_12, hits1_12, mrr_12

