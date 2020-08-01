from openea.modules.load.kg import KG
from openea.modules.load.read import *
import os, sys
import copy

class KGs:
    def __init__(self, kg1: KG, kg2: KG, train_links, test_links, valid_links=None, rel_links=None, kg_test=None, kg_valid=None, mode='mapping', ordered=True, training_data_folder=None, train_kg='kg12'):

        self.uri_kg1, self.uri_kg2 = kg1 , kg2
        self.uri_kg_test, self.uri_kg_valid = kg_test, kg_valid

        self.ent_ids1, self.rel_ids1, self.ent_ids2, self.rel_ids2 = self.generate_ent_rel_ids(kg1, kg2,
                    train_links, rel_links, kg_test, kg_valid, mode, ordered, training_data_folder, train_kg)
        self.kg1 = None
        self.kg2= None

        if train_kg=="kg1":
            self.kg1 = self.init_kg(kg1, self.ent_ids1, self.rel_ids1)
            self.kg1_test = self.init_kg(kg_test, self.ent_ids1, self.rel_ids1)
            self.kg1_valid = self.init_kg(kg_valid, self.ent_ids1, self.rel_ids1)
            self.create_filter_rank_dict(self.kg1.relation_triples_list, self.kg1_test, self.kg1_valid)

            self.relations_classes = self.kg1.relations_classes
            self.entities_num = len(self.ent_ids1.values())
            self.relations_num = len(self.rel_ids1.values())

            self.rel_ids_mask = self.kg1.id_relations_dict
            self.relations_mask_num = len(self.rel_ids_mask.values())

            assert self.entities_num == len(self.kg1.entities_set |  self.kg1_test.entities_set | self.kg1_valid.entities_set)
            assert self.relations_num == len(self.kg1.relations_set | self.kg1_test.relations_set | self.kg1_valid.relations_set)
            print("All entities_num: {},  kg1: {}, kg_test: {}, kg_valid: {}".format(self.entities_num, len(self.kg1.entities_set),
                                                                len(self.kg1_test.entities_set) , len(self.kg1_valid.entities_set)))

        elif train_kg=="kg2":
            self.kg2 = self.init_kg(kg2, self.ent_ids2, self.rel_ids2)
            self.kg2_test = self.init_kg(kg_test, self.ent_ids2, self.rel_ids2)
            self.kg2_valid = self.init_kg(kg_valid, self.ent_ids2, self.rel_ids2)
            self.create_filter_rank_dict(self.kg2.relation_triples_list, self.kg2_test, self.kg2_valid)

            self.relations_classes = self.kg2.relations_classes
            self.entities_num = len(self.ent_ids2.values())
            self.relations_num = len(self.rel_ids2.values())

            rel_ids_mask = copy.deepcopy(self.kg2.relations_id_dict)
            del rel_ids_mask["FW-REL"]
            self.rel_ids_mask  = {v:k for k, v in rel_ids_mask.items()}
            self.relations_mask_num = len(self.rel_ids_mask.values())

            assert self.entities_num == len(self.kg2.entities_set |  self.kg2_test.entities_set | self.kg2_valid.entities_set)
            assert self.relations_num == len(self.kg2.relations_set | self.kg2_test.relations_set | self.kg2_valid.relations_set)
            print("All entities_num: {},  kg2: {}, kg_test: {}, kg_valid: {}".format(self.entities_num, len(self.kg2.entities_set),
                                                                len(self.kg2_test.entities_set) , len(self.kg2_valid.entities_set)))

        elif train_kg=="kg12":
            self.kg1 = self.init_kg(kg1, self.ent_ids1, self.rel_ids1)
            self.kg2 = self.init_kg(kg2, self.ent_ids2, self.rel_ids2)

            #??? choose which one to for evaluation? maybe both?
            self.kg1_test = self.init_kg(kg_test, self.ent_ids1, self.rel_ids1)
            self.kg1_valid = self.init_kg(kg_valid, self.ent_ids1, self.rel_ids1)
            self.create_filter_rank_dict(self.kg1.relation_triples_list, self.kg1_test, self.kg1_valid)

            self.kg2_test = self.init_kg(kg_test, self.ent_ids2, self.rel_ids2)
            self.kg2_valid = self.init_kg(kg_valid, self.ent_ids2, self.rel_ids2)
            self.create_filter_rank_dict(self.kg2.relation_triples_list, self.kg2_test, self.kg2_valid)

            self.relations_classes = self.kg1.relations_classes
            self.entities_num = len(self.kg1.entities_set | self.kg2.entities_set | self.kg1_test.entities_set | self.kg1_valid.entities_set)
            self.relations_num = len(self.kg1.relations_set | self.kg2.relations_set | self.kg1_test.relations_set | self.kg1_valid.relations_set)

            self.rel_ids_mask = self.kg1.id_relations_dict
            self.relations_mask_num = len(self.rel_ids_mask.values())

            self.init_align_links(train_links, test_links, valid_links, mode)

            print("All entities_num: {}, kg1: {}, kg2: {}, kg1_test: {}, kg1_valid: {}".format(self.entities_num, len(self.kg1.entities_set),
                                            len(self.kg2.entities_set), len(self.kg1_test.entities_set) , len(self.kg2_valid.entities_set)))

        #sys.exit()

    def generate_ent_rel_ids(self, kg1: KG, kg2: KG, train_links, rel_links=None, kg_test=None, kg_valid=None, mode="mapping", ordered=True, training_data_folder=None, train_kg='kg12'):

        kg1_useful_triples_set = kg1.relation_triples_set | kg_test.relation_triples_set | kg_valid.relation_triples_set
        kg1_useful_entities_set = kg1.entities_set | kg_test.entities_set | kg_valid.entities_set
        kg1_useful_relations_set = kg1.relations_set | kg_test.relations_set | kg_valid.relations_set

        kg2_useful_triples_set = kg2.relation_triples_set | kg_test.relation_triples_set | kg_valid.relation_triples_set
        kg2_useful_entities_set = kg2.entities_set| kg_test.entities_set | kg_valid.entities_set
        kg2_useful_relations_set = kg2.relations_set | kg_test.relations_set | kg_valid.relations_set

        ent_ids1, rel_ids1, ent_ids2, rel_ids2 = {}, {}, {}, {}

        if train_kg=="kg1":
            ent_ids1= generate_id(kg1_useful_triples_set, kg1_useful_entities_set, ordered=ordered)
            rel_ids1= generate_id(kg1_useful_triples_set, kg1_useful_relations_set, ordered=False)
            print("Relations and ids: {}".format(rel_ids1))

        elif train_kg=="kg2":
            ent_ids2 = generate_id(kg2_useful_triples_set, kg2_useful_entities_set, ordered=ordered)
            rel_ids2 = generate_id(kg2_useful_triples_set, kg2_useful_relations_set, ordered=False)
            print("Relations and ids: {}".format(rel_ids2))

        elif train_kg=="kg12":
            if mode == "sharing":
                ent_ids1, ent_ids2 = generate_sharing_id(train_links, kg1_useful_triples_set, kg1_useful_entities_set,
                                                    kg2_useful_triples_set, kg2_useful_entities_set, ordered=ordered)

                rel_ids1, rel_ids2 = generate_sharing_id(rel_links, kg1_useful_triples_set, kg1_useful_relations_set,
                                                        kg2_useful_triples_set, kg2_useful_relations_set, ordered=False)

            else:
                ent_ids1, ent_ids2 = generate_mapping_id(kg1_useful_triples_set, kg1_useful_entities_set,
                                        kg2_useful_triples_set, kg2_useful_entities_set, ordered=ordered)
                #rel_ids1, rel_ids2 = generate_mapping_id(kg1_useful_triples_set, kg1_useful_relations_set,
                #                        kg2_useful_triples_set, kg2_useful_relations_set, ordered=False)
                rel_ids1, rel_ids2 = generate_sharing_id(rel_links, kg1_useful_triples_set, kg1_useful_relations_set, # share relations anyway
                                                        kg2_useful_triples_set, kg2_useful_relations_set, ordered=False)

            print("Relations and ids: {} {} ".format(rel_ids1, rel_ids2))

        return ent_ids1, rel_ids1, ent_ids2, rel_ids2

    def init_kg(self, uri_kg, ent_ids, rel_ids):
        id_relation_triples = uris_relation_triple_2ids(uri_kg.relation_triples_set, ent_ids, rel_ids)
        kg = KG(id_relation_triples)
        kg.set_id_dict(ent_ids, rel_ids)
        return kg

    def create_filter_rank_dict(self, id_relation_triples, kg_test, kg_valid):
        #For filtering rank
        self.set_multi_entities_dict(id_relation_triples + kg_test.relation_triples_list + kg_valid.relation_triples_list)
        kg_valid.set_local_multi_entities_dict(self.hr_to_multi_t, self.tr_to_multi_h)
        kg_test.set_local_multi_entities_dict(self.hr_to_multi_t, self.tr_to_multi_h)

    def init_align_links(self, train_links, test_links, valid_links, mode):
        #For alignment evaluation
        self.uri_train_links = train_links
        self.uri_test_links = test_links
        self.train_links = uris_pair_2ids(self.uri_train_links, self.ent_ids1, self.ent_ids2)
        self.test_links = uris_pair_2ids(self.uri_test_links, self.ent_ids1, self.ent_ids2)

        self.train_entities1 = [link[0] for link in self.train_links]
        self.train_entities2 = [link[1] for link in self.train_links]
        self.test_entities1 = [link[0] for link in self.test_links]
        self.test_entities2 = [link[1] for link in self.test_links]

        self.valid_links = list()
        self.valid_entities1 = list()
        self.valid_entities2 = list()
        if valid_links is not None:
            self.uri_valid_links = valid_links
            self.valid_links = uris_pair_2ids(self.uri_valid_links, self.ent_ids1, self.ent_ids2)
            self.valid_entities1 = [link[0] for link in self.valid_links]
            self.valid_entities2 = [link[1] for link in self.valid_links]

        if mode == 'swapping':
            self.add_swapped_links()

        self.useful_entities_list1 = self.train_entities1 + self.valid_entities1 + self.test_entities1
        self.useful_entities_list2 = self.train_entities2 + self.valid_entities2 + self.test_entities2

    def add_swapped_links(self):
        sup_triples1, sup_triples2 = generate_sup_relation_triples(self.train_links,
                                                                    self.kg1.rt_dict, self.kg1.hr_dict,
                                                                    self.kg2.rt_dict, self.kg2.hr_dict)
        self.kg1.add_sup_relation_triples(sup_triples1)
        self.kg2.add_sup_relation_triples(sup_triples2)



    def set_multi_entities_dict(self, relation_triples_list):
        '''filter for evaluating on local graph'''
        self.hr_to_multi_t, self.tr_to_multi_h= dict(), dict()

        for h, r, t in relation_triples_list:
            t_set = self.hr_to_multi_t.get((h,r), set())
            t_set.add(t)
            self.hr_to_multi_t[(h,r)] = t_set

            h_set = self.tr_to_multi_h.get((t,r), set())
            h_set.add((h))
            self.tr_to_multi_h[(t,r)] = h_set




def read_kgs_from_folder(training_data_folder, division, mode, ordered, remove_unlinked=False, exist_attr=True, train_kg="kg12"):
    if 'dbp15k' in training_data_folder.lower() or 'dwy100k' in training_data_folder.lower():
        return read_kgs_from_dbp_dwy(training_data_folder, division, mode, ordered, remove_unlinked=remove_unlinked)
    kg1_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_1')
    kg2_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_2')

    test_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_test')
    valid_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_valid')

    if exist_attr:
        kg1_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_1')
        kg2_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_2')

    train_links = read_links(training_data_folder + 'ent_links_train')
    valid_links = read_links(training_data_folder + 'ent_links_valid')
    test_links = read_links(training_data_folder + 'ent_links_test')
    rel_links = read_links(training_data_folder + 'rel_links')

    if remove_unlinked:
        links = train_links + valid_links + test_links
        kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
        kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)

    if exist_attr:
        kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
        kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
        kgs = KGs(kg1, kg2, train_links, test_links, valid_links=valid_links, mode=mode, ordered=ordered)
    else:
        kg1 = KG(kg1_relation_triples)
        kg2 = KG(kg2_relation_triples)
        kg_test= KG(test_relation_triples)
        kg_valid = KG(valid_relation_triples)

        kgs = KGs(kg1, kg2, train_links, test_links, valid_links=valid_links, rel_links=rel_links, kg_test=kg_test, kg_valid=kg_valid, mode=mode, ordered=ordered, training_data_folder=training_data_folder, train_kg=train_kg)
    return kgs


def read_reversed_kgs_from_folder(training_data_folder, division, mode, ordered, remove_unlinked=False):
    kg1_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_2')
    kg2_relation_triples, _, _ = read_relation_triples(training_data_folder + 'rel_triples_1')
    kg1_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_2')
    kg2_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_1')

    temp_train_links = read_links(training_data_folder + division + 'train_links')
    temp_valid_links = read_links(training_data_folder + division + 'valid_links')
    temp_test_links = read_links(training_data_folder + division + 'test_links')
    train_links = [(j, i) for i, j in temp_train_links]
    valid_links = [(j, i) for i, j in temp_valid_links]
    test_links = [(j, i) for i, j in temp_test_links]

    if remove_unlinked:
        links = train_links + valid_links + test_links
        kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
        kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)

    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
    kgs = KGs(kg1, kg2, train_links, test_links, valid_links=valid_links, mode=mode, ordered=ordered)
    return kgs


def read_kgs_from_files(kg1_relation_triples, kg2_relation_triples, kg1_attribute_triples, kg2_attribute_triples,
                        train_links, valid_links, test_links, mode):
    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
    kgs = KGs(kg1, kg2, train_links, test_links, valid_links=valid_links, mode=mode)
    return kgs


def read_kgs_from_dbp_dwy(folder, division, mode, ordered, remove_unlinked=False):
    folder = folder + division
    kg1_relation_triples, _, _ = read_relation_triples(folder + 'triples_1')
    kg2_relation_triples, _, _ = read_relation_triples(folder + 'triples_2')
    if os.path.exists(folder + 'sup_pairs'):
        train_links = read_links(folder + 'sup_pairs')
    else:
        train_links = read_links(folder + 'sup_ent_ids')
    if os.path.exists(folder + 'ref_pairs'):
        test_links = read_links(folder + 'ref_pairs')
    else:
        test_links = read_links(folder + 'ref_ent_ids')
    print()
    if remove_unlinked:
        for i in range(10000):
            print("removing times:", i)
            links = train_links + test_links
            kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
            kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)
            n1 = len(kg1_relation_triples)
            n2 = len(kg2_relation_triples)
            train_links, test_links = remove_no_triples_link(kg1_relation_triples, kg2_relation_triples,
                                                             train_links, test_links)
            links = train_links + test_links
            kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
            kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)
            n11 = len(kg1_relation_triples)
            n22 = len(kg2_relation_triples)
            if n1 == n11 and n2 == n22:
                break
            print()

    kg1 = KG(kg1_relation_triples, list())
    kg2 = KG(kg2_relation_triples, list())
    kgs = KGs(kg1, kg2, train_links, test_links, mode=mode, ordered=ordered)
    return kgs


def remove_no_triples_link(kg1_relation_triples, kg2_relation_triples, train_links, test_links):
    kg1_entities, kg2_entities = set(), set()
    for h, r, t in kg1_relation_triples:
        kg1_entities.add(h)
        kg1_entities.add(t)
    for h, r, t in kg2_relation_triples:
        kg2_entities.add(h)
        kg2_entities.add(t)
    print("before removing links with no triples:", len(train_links), len(test_links))
    new_train_links, new_test_links = set(), set()
    for i, j in train_links:
        if i in kg1_entities and j in kg2_entities:
            new_train_links.add((i, j))
    for i, j in test_links:
        if i in kg1_entities and j in kg2_entities:
            new_test_links.add((i, j))
    print("after removing links with no triples:", len(new_train_links), len(new_test_links))
    return list(new_train_links), list(new_test_links)


def remove_unlinked_triples(triples, links):
    print("before removing unlinked triples:", len(triples))
    linked_entities = set()
    for i, j in links:
        linked_entities.add(i)
        linked_entities.add(j)
    linked_triples = set()
    for h, r, t in triples:
        if h in linked_entities and t in linked_entities:
            linked_triples.add((h, r, t))
    print("after removing unlinked triples:", len(linked_triples))
    return linked_triples


def write_kg1_to_files(out_folder, ent_ids, rel_ids, id_triples_train, id_triples_valid, id_triples_test ):
    '''#write_kg1_to_files(training_data_folder+'kg1/', ent_ids1, rel_ids1,
        #                    self.uri_kg1.relation_triples_list, self.uri_kg_valid.relation_triples_list,
        #                    self.uri_kg_test.relation_triples_list)

        #sys.exit()

    # entity2id.txt: ent_ids1
    # relation2id.txt rel_ids1
    # train2id.txt id_relation_triples1
    # test2id_all.txt kg_test.relation_triples_list
    # valid2id.txt kg_valid.relation_triples_list
    '''
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    dict2file(out_folder+'entity2id.txt', ent_ids)
    dict2file(out_folder+'relation2id.txt', rel_ids)
    triple2file(out_folder+'train.txt', id_triples_train, 'htr')
    triple2file(out_folder+'valid.txt', id_triples_valid, 'htr')
    triple2file(out_folder+'test.txt', id_triples_test, 'htr')
