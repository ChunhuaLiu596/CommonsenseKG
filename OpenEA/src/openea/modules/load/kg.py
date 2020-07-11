def parse_triples(triples):
    subjects, predicates, objects = set(), set(), set()
    for s, p, o in triples:
        subjects.add(s)
        predicates.add(p)
        objects.add(o)
    return subjects, predicates, objects


class KG:
    def __init__(self, relation_triples, attribute_triples=None):

        self.entities_set, self.entities_list = None, None
        self.relations_set, self.relations_list = None, None
        self.attributes_set, self.attributes_list = None, None
        self.entities_num, self.relations_num, self.attributes_num = None, None, None
        self.relation_triples_num, self.attribute_triples_num = None, None
        self.local_relation_triples_num, self.local_attribute_triples_num = None, None

        self.entities_id_dict = None
        self.id_entities_dict = None
        self.relations_id_dict = None
        self.id_relations_dict = None
        self.attributes_id_dict = None

        self.rt_dict, self.hr_dict = None, None
        self.entity_relations_dict = None
        self.entity_attributes_dict = None
        self.av_dict = None

        self.sup_relation_triples_set, self.sup_relation_triples_list = None, None
        self.sup_attribute_triples_set, self.sup_attribute_triples_list = None, None

        self.relation_triples_set = None
        self.attribute_triples_set = None
        self.relation_triples_list = None
        self.attribute_triples_list = None

        self.relation_triples_set = None
        self.relation_triples_list = None
        self.local_attribute_triples_set = None
        self.local_attribute_triples_list = None

        self.rel_classes = None

        self.set_relations(relation_triples)
        if attribute_triples is not None:
            self.set_attributes(attribute_triples)

        print()
        print("KG: statistics:")
        print("Number of entities:", self.entities_num)
        print("Number of relations:", self.relations_num)
        print("Number of attributes:", self.attributes_num)
        print("Number of relation triples:", self.relation_triples_num)
        print("Number of attribute triples:", self.attribute_triples_num)
        print("Number of local relation triples:", self.local_relation_triples_num)
        print("Number of local attribute triples:", self.local_attribute_triples_num)
        print()

    def set_relations(self, relation_triples):
        self.relation_triples_set = set(relation_triples)
        self.relation_triples_list = list(self.relation_triples_set)
        self.relation_triples_set = self.relation_triples_set
        self.relation_triples_list = self.relation_triples_list

        heads, relations, tails = parse_triples(self.relation_triples_set)
        self.entities_set = heads | tails
        self.relations_set = relations
        self.entities_list = list(self.entities_set)
        self.relations_list = list(self.relations_set)
        self.entities_num = len(self.entities_set)
        self.relations_num = len(self.relations_set)
        self.relation_triples_num = len(self.relation_triples_set)
        self.local_relation_triples_num = len(self.relation_triples_set)
        self.generate_relation_triple_dict()
        self.parse_relations()

    def set_attributes(self, attribute_triples):
        self.attribute_triples_set = set(attribute_triples)
        self.attribute_triples_list = list(self.attribute_triples_set)
        self.local_attribute_triples_set = self.attribute_triples_set
        self.local_attribute_triples_list = self.attribute_triples_list

        entities, attributes, values = parse_triples(self.attribute_triples_set)
        self.attributes_set = attributes
        self.attributes_list = list(self.attributes_set)
        self.attributes_num = len(self.attributes_set)

        self.attribute_triples_num = len(self.attribute_triples_set)
        self.local_attribute_triples_num = len(self.local_attribute_triples_set)
        self.generate_attribute_triple_dict()
        self.parse_attributes()

    def generate_relation_triple_dict(self):
        self.rt_dict, self.hr_dict = dict(), dict()
        for h, r, t in self.relation_triples_list:
            rt_set = self.rt_dict.get(h, set())
            rt_set.add((r, t))
            self.rt_dict[h] = rt_set
            hr_set = self.hr_dict.get(t, set())
            hr_set.add((h, r))
            self.hr_dict[t] = hr_set
        print("Number of rt_dict:", len(self.rt_dict))
        print("Number of hr_dict:", len(self.hr_dict))


    def generate_attribute_triple_dict(self):
        self.av_dict = dict()
        for h, a, v in self.local_attribute_triples_list:
            av_set = self.av_dict.get(h, set())
            av_set.add((a, v))
            self.av_dict[h] = av_set
        print("Number of av_dict:", len(self.av_dict))

    def parse_relations(self):
        self.entity_relations_dict = dict()
        for ent, attr, _ in self.relation_triples_set:
            attrs = self.entity_relations_dict.get(ent, set())
            attrs.add(attr)
            self.entity_relations_dict[ent] = attrs
        print("entity relations dict:", len(self.entity_relations_dict))

    def parse_attributes(self):
        self.entity_attributes_dict = dict()
        for ent, attr, _ in self.local_attribute_triples_set:
            attrs = self.entity_attributes_dict.get(ent, set())
            attrs.add(attr)
            self.entity_attributes_dict[ent] = attrs
        print("entity attributes dict:", len(self.entity_attributes_dict))

    def set_id_dict(self, entities_id_dict, relations_id_dict, attributes_id_dict=None):
        self.entities_id_dict = entities_id_dict
        self.id_entities_dict = {v:k for k, v in entities_id_dict.items()}
        self.relations_id_dict = relations_id_dict
        self.id_relations_dict = {v:k for k, v in relations_id_dict.items()}
        if attributes_id_dict is not None:
            self.attributes_id_dict = attributes_id_dict

        '''convert the id to rel name for drawing confusion matrix'''
        self.relations_classes = [ self.id_relations_dict[rid] for rid  in self.relations_list]

        self.set_local_id_dict()
        self.set_local_id_triples()

    def add_sup_relation_triples(self, sup_triples):
        self.sup_relation_triples_set = set(sup_triples)
        self.sup_relation_triples_list = list(self.sup_relation_triples_set)
        self.relation_triples_set |= sup_triples
        self.relation_triples_list = list(self.relation_triples_set)
        self.relation_triples_num = len(self.relation_triples_list)

    def add_sup_attribute_triples(self, sup_triples):
        self.sup_attribute_triples_set = set(sup_triples)
        self.sup_attribute_triples_list = list(self.sup_attribute_triples_set)
        self.attribute_triples_set |= sup_triples
        self.attribute_triples_list = list(self.attribute_triples_set)
        self.attribute_triples_num = len(self.attribute_triples_list)

    #def set_rel_classes(self):
    #    '''convert the id to rel name'''
    #    self.relations_classes = [ self.id_relations_dict[rid] for rid  in self.relations_list]

    def get_rel_classes(self, ids, id_relations_classes):
        classes = [id_relations_classes[idx] for idx in ids]
        return classes

    def set_local_id_dict(self):
        '''
        Function: For compute rank on local graph
        'global*' means the id dict is computed based all all data (e.g., train+valid+test)
        'local*' means the id dict is computed based the sub-data
        '''

        self.local_id_entities_dict = dict()
        self.global_local_ids_dict = dict()
        for local_id, global_id in enumerate(self.entities_list):
            ent = self.id_entities_dict.get(global_id)
            self.local_id_entities_dict[local_id] = ent
            self.global_local_ids_dict[global_id]= local_id

    def set_local_id_triples(self):
        '''Golden rank for evaluating on local graph'''
        self.local_relation_triples_list = list()
        for (h,r,t) in self.relation_triples_list:
            triple = (self.global_local_ids_dict[h], r, self.global_local_ids_dict[t])
            self.local_relation_triples_list.append(triple)

    def set_local_multi_entities_dict(self, hr_to_multi_t, tr_to_multi_h):
        '''filter for evaluating on local graph'''

        def map_global_dic_to_local(dic):
            local_id_dic = {}
            for (e1, rel), e2s in dic.items():
                for e2 in e2s:
                    if e1 in self.global_local_ids_dict and e2 in self.global_local_ids_dict:
                        local_id_e1 = self.global_local_ids_dict[e1]
                        local_id_e2 = self.global_local_ids_dict[e2]

                        if (local_id_e1, rel) in local_id_dic:
                            local_id_dic[(local_id_e1, rel)].append(local_id_e2)
                        else:
                            local_id_dic[(local_id_e1, rel)] = [local_id_e2]
            return local_id_dic

        self.local_hr_to_multi_t = map_global_dic_to_local(hr_to_multi_t)
        self.local_tr_to_multi_h = map_global_dic_to_local(tr_to_multi_h)

                #print("Number of rt_dict:", len(self.rt_dict))
        #print("Number of hr_dict:", len(self.hr_dict))



