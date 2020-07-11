import os

import numpy as np
import pandas as pd


def load_embeddings(file_name):
    if os.path.exists(file_name):
        return np.load(file_name)
    return None


def sort_elements(triples, elements_set):
    dic = dict()
    for s, p, o in triples:
        if s in elements_set:
            dic[s] = dic.get(s, 0) + 1
        if p in elements_set:
            dic[p] = dic.get(p, 0) + 1
        if o in elements_set:
            dic[o] = dic.get(o, 0) + 1
    # firstly sort by values (i.e., frequencies), if equal, by keys (i.e, URIs)
    sorted_list = sorted(dic.items(), key=lambda x: (x[1], x[0]), reverse=True)
    ordered_elements = [x[0] for x in sorted_list]
    return ordered_elements, dic


def generate_sharing_id(train_links, kg1_triples, kg1_elements, kg2_triples, kg2_elements, ordered=True):
    ids1, ids2 = dict(), dict()
    if ordered:
        linked_dic = dict()
        for x, y in train_links:
            linked_dic[y] = x
        kg2_linked_elements = [x[1] for x in train_links]
        kg2_unlinked_elements = set(kg2_elements) - set(kg2_linked_elements)
        ids1, ids2 = generate_mapping_id(kg1_triples, kg1_elements, kg2_triples, kg2_unlinked_elements, ordered=ordered)
        for ele in kg2_linked_elements:
            ids2[ele] = ids1[linked_dic[ele]]
    else:
        index = 0
        for e1, e2 in train_links:
            assert e1 in kg1_elements
            assert e2 in kg2_elements
            ids1[e1] = index
            ids2[e2] = index
            index += 1
        for ele in kg1_elements:
            if ele not in ids1:
                ids1[ele] = index
                index += 1
        for ele in kg2_elements:
            if ele not in ids2:
                ids2[ele] = index
                index += 1
    assert len(ids1) == len(set(kg1_elements))
    assert len(ids2) == len(set(kg2_elements))
    return ids1, ids2


def generate_mapping_id(kg1_triples, kg1_elements, kg2_triples, kg2_elements, ordered=True):
    ids1, ids2 = dict(), dict()
    if ordered:
        kg1_ordered_elements, _ = sort_elements(kg1_triples, kg1_elements)
        kg2_ordered_elements, _ = sort_elements(kg2_triples, kg2_elements)
        n1 = len(kg1_ordered_elements)
        n2 = len(kg2_ordered_elements)
        n = max(n1, n2)
        for i in range(n):
            if i < n1 and i < n2:
                ids1[kg1_ordered_elements[i]] = i * 2
                ids2[kg2_ordered_elements[i]] = i * 2 + 1
            elif i >= n1:
                ids2[kg2_ordered_elements[i]] = n1 * 2 + (i - n1)
            else:
                ids1[kg1_ordered_elements[i]] = n2 * 2 + (i - n2)
    else:
        index = 0
        for ele in kg1_elements:
            if ele not in ids1:
                ids1[ele] = index
                index += 1
        for ele in kg2_elements:
            if ele not in ids2:
                ids2[ele] = index
                index += 1
    assert len(ids1) == len(set(kg1_elements))
    assert len(ids2) == len(set(kg2_elements))
    return ids1, ids2

def generate_id(kg1_triples, kg1_elements, ordered=True):
    ids1  = dict()
    if ordered:
        kg1_ordered_elements, _ = sort_elements(kg1_triples, kg1_elements)
        n = len(kg1_ordered_elements)
        for i in range(n):
            ids1[kg1_ordered_elements[i]] = i
    else:
        index = 0
        for ele in kg1_elements:
            if ele not in ids1:
                ids1[ele] = index
                index += 1
    assert len(ids1) == len(set(kg1_elements))
    return ids1


def uris_list_2ids(uris, ids):
    id_uris = list()
    for u in uris:
        assert u in ids
        id_uris.append(ids[u])
    assert len(id_uris) == len(set(uris))
    return id_uris


def uris_pair_2ids(uris, ids1, ids2):
    id_uris = list()
    for u1, u2 in uris:
        # assert u1 in ids1
        # assert u2 in ids2
        if u1 in ids1 and u2 in ids2:
            id_uris.append((ids1[u1], ids2[u2]))
        else:
            print ("{} {} ".format(u1, u2))
    # assert len(id_uris) == len(set(uris))
    return id_uris


def uris_relation_triple_2ids(uris, ent_ids, rel_ids):
    id_uris = list()
    for u1, u2, u3 in uris:
        assert u1 in ent_ids
        assert u2 in rel_ids
        assert u3 in ent_ids
        id_uris.append((ent_ids[u1], rel_ids[u2], ent_ids[u3]))
    assert len(id_uris) == len(set(uris))
    return id_uris


def uris_attribute_triple_2ids(uris, ent_ids, attr_ids):
    id_uris = list()
    for u1, u2, u3 in uris:
        assert u1 in ent_ids
        assert u2 in attr_ids
        id_uris.append((ent_ids[u1], attr_ids[u2], u3))
    assert len(id_uris) == len(set(uris))
    return id_uris


def generate_sup_relation_triples_one_link(e1, e2, rt_dict, hr_dict):
    new_triples = set()
    for r, t in rt_dict.get(e1, set()):
        new_triples.add((e2, r, t))
    for h, r in hr_dict.get(e1, set()):
        new_triples.add((h, r, e2))
    return new_triples


def generate_sup_relation_triples(sup_links, rt_dict1, hr_dict1, rt_dict2, hr_dict2):
    new_triples1, new_triples2 = set(), set()
    for ent1, ent2 in sup_links:
        new_triples1 |= (generate_sup_relation_triples_one_link(ent1, ent2, rt_dict1, hr_dict1))
        new_triples2 |= (generate_sup_relation_triples_one_link(ent2, ent1, rt_dict2, hr_dict2))
    print("supervised relation triples: {}, {}".format(len(new_triples1), len(new_triples2)))
    return new_triples1, new_triples2


def generate_sup_attribute_triples_one_link(e1, e2, av_dict):
    new_triples = set()
    for a, v in av_dict.get(e1, set()):
        new_triples.add((e2, a, v))
    return new_triples


def generate_sup_attribute_triples(sup_links, av_dict1, av_dict2):
    new_triples1, new_triples2 = set(), set()
    for ent1, ent2 in sup_links:
        new_triples1 |= (generate_sup_attribute_triples_one_link(ent1, ent2, av_dict1))
        new_triples2 |= (generate_sup_attribute_triples_one_link(ent2, ent1, av_dict2))
    print("supervised attribute triples: {}, {}".format(len(new_triples1), len(new_triples2)))
    return new_triples1, new_triples2


def read_relation_triples(file_path, use_rel_subset=False):
    if use_rel_subset:
        rel_subset=['AtLocation', 'IsA']
    else:
        rel_subset=None

    if file_path is None:
        return set(), set(), set()

    sample=False
    triples = set()
    entities, relations = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 3
        h = params[0].strip()
        r = params[1].strip()
        t = params[2].strip()

        if use_rel_subset and r not in rel_subset:
            sample=False
        else:
            sample=True

        if sample:
            triples.add((h, r, t))
            entities.add(h)
            entities.add(t)
            relations.add(r)
    print("read {} relation triples from {} ".format(len(triples), file_path))
    return triples, entities, relations


def read_links(file_path):
    print("read links:", file_path)
    links = list()
    refs = list()
    reft = list()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        e1 = params[0].strip()
        e2 = params[1].strip()
        refs.append(e1)
        reft.append(e2)
        links.append((e1, e2))
    assert len(refs) == len(reft)
    return links


def read_dict(file_path):
    file = open(file_path, 'r', encoding='utf8')
    ids = dict()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        ids[params[0]] = int(params[1])
    file.close()
    return ids


def read_pair_ids(file_path):
    file = open(file_path, 'r', encoding='utf8')
    pairs = list()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        pairs.append((int(params[0]), int(params[1])))
    file.close()
    return pairs

def triple2file(file, triples, mode='hrt'):
    '''triples: h,r,t '''

    if triples is None:
        return
    if mode=='hrt':
        with open(file, 'w', encoding='utf8') as f:
            for i, j, k in triples:
                f.write(str(i) + '\t' + str(j) + '\t' + str(k) + '\n')
            f.close()
    elif mode=='htr':
        with open(file, 'w', encoding='utf8') as f:
            for i, j, k in triples:
                f.write(str(i) + '\t' + str(k) + '\t' + str(j) + '\n')
            f.close()
    print(file, "saved.")

def pair2csv(file, pairs):
    if pairs is None:
        return
    lines=list()
    for i,  (x, y) in enumerate(pairs):
        #vessel	[('vessel', 0.6274), ('tube', 0.5497), ('tank', 0.5049), ('shell', 0.5025), ('dish', 0.4986), ('plastic', 0.4788), ('bottle', 0.4788), ('sailboat', 0.4748), ('liquid', 0.4723), ('soap', 0.4609)]
        line= [(x)]
        for y1, y2 in y:
            line.extend([y1])
            line.extend([y2])
        lines.append(line)

    columns = ["goldens"]
    for i in range(1,11):
        columns.append("rank-{}".format(i))
        columns.append("score-{}".format(i))
    df=pd.DataFrame(lines, columns=columns)
    df.to_csv(file+".csv", index=True)
    print(file, "saved.")


def pair2file(file, pairs):
    if pairs is None:
        return
    with open(file, 'w', encoding='utf8') as f:
        for i, j in pairs:
            out = "{}\t{}\n".format(i,j)
            f.write(out)
        f.close()
    print("save", file)


def dict2file(file, dic, sort_value=True):
    if dic is None:
        print("dic doesn't exist")
        return

    if sort_value:
        dic = sorted(dic.items(), key=lambda x: x[1], reverse=False)
        with open(file, 'w', encoding='utf8') as f:
                for x in dic:
                    f.write(str(x[0]) + '\t' + str(x[1]) + '\n')
                f.close()
    else:
        with open(file, 'w', encoding='utf8') as f:
            for i, j in dic.items():
                f.write(str(i) + '\t' + str(j) + '\n')
            f.close()
    print(file, "saved.")


def line2file(file, lines):
    if lines is None:
        return
    with open(file, 'w', encoding='utf8') as f:
        for line in lines:
            f.write(line + '\n')
        f.close()
    print(file, "saved.")


def radio_2file(radio, folder):
    path = folder + str(radio).replace('.', '_')
    if not os.path.exists(path):
        os.makedirs(path)
    return path + '/'


def save_results(folder, rest_12, save_name='alignment_results_12'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    #pair2file(folder + save_name , rest_12)
    pair2csv(folder + save_name , rest_12)
    print("Results {} saved!".format(save_name))


def save_embeddings(folder, kgs, ent_embeds, rel_embeds, attr_embeds, mapping_mat=None, rev_mapping_mat=None):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if ent_embeds is not None:
        #np.save(folder + 'ent_embeds.npy', ent_embeds)
        np.savetxt(folder + "ent_embeds.tsv", ent_embeds, delimiter="\t")
    if rel_embeds is not None:
        #np.save(folder + 'rel_embeds.npy', rel_embeds)
        np.savetxt(folder + 'rel_embeds.tsv', rel_embeds, delimiter="\t")
    if attr_embeds is not None:
        np.save(folder + 'attr_embeds.npy', attr_embeds)
    if mapping_mat is not None:
        np.save(folder + 'mapping_mat.npy', mapping_mat)
    if rev_mapping_mat is not None:
        np.save(folder + 'rev_mapping_mat.npy', rev_mapping_mat)

    if kgs.kg1 is not None and kgs.kg2 is not None:
        dict2file(folder + 'kg12_ent_ids', {**kgs.kg1.entities_id_dict, **kgs.kg2.entities_id_dict})
    if kgs.kg1 is not None:
        dict2file(folder + 'kg1_ent_ids', kgs.kg1.entities_id_dict)
        dict2file(folder + 'kg1_rel_ids', kgs.kg1.relations_id_dict)
    if kgs.kg2 is not None:
        dict2file(folder + 'kg2_ent_ids', kgs.kg2.entities_id_dict)
        dict2file(folder + 'kg2_rel_ids', kgs.kg2.relations_id_dict)
    print("Embeddings saved!")


def read_attribute_triples(file_path):
    print("read attribute triples:", file_path)
    if file_path is None:
        return set(), set(), set()
    if file_path is None:
        return set(), set(), set()
    triples = set()
    entities, attributes = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip().strip('\n').split('\t')
        if len(params) < 3:
            continue
        head = params[0].strip()
        attr = params[1].strip()
        value = params[2].strip()
        if len(params) > 3:
            for p in params[3:]:
                value = value + ' ' + p.strip()
        value = value.strip().rstrip('.').strip()
        entities.add(head)
        attributes.add(attr)
        triples.add((head, attr, value))
    return triples, entities, attributes


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



def write_rel_score_to_file(gold_scores, pred_scores, out_folder):
    rel_out = list()
    for i, (gold_rel, pred_rel) in enumerate(zip(gold_scores, pred_scores)):
        gold_rel_id = np.argmax(gold_rel)
        pred_rel_id = np.argmax(pred_rel)
        score = pred_rel
        rel_out.append(((gold_rel_id, pred_rel_id),score))
    pair2file(out_folder+"relation_pred_score", rel_out)

if __name__ == '__main__':
    mydict = {'b': 10, 'c': 10, 'a': 10, 'd': 20}
    sorted_dic = sorted(mydict.items(), key=lambda x: (x[1], x[0]), reverse=True)
    print(sorted_dic, type(sorted_dic))
