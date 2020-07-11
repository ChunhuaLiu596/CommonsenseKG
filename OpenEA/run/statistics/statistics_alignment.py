from collections import Counter
import argparse
import numpy as np
import sys
import os
import json
import time
import random
import pdb
import datetime
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import graph
from reader import  ConceptNetTSVReader, SwowTSVReader, Triple2GraphReader
from utils_writer import * 
import logging as logger
logger.getLogger().setLevel(logger.INFO)

class Alignment(object):
    def __init__(self, args):
        self.sw_net = self.build_net(args.swow_source_file, SwowTSVReader)
        self.cn_net = self.build_net(args.conceptnet_source_file, ConceptNetTSVReader)
        
        self.overlap_edges_set, self.overlap_edges_nodes, self.overlap_rels= self.find_overlap_edges()
        self.overlap_nodes = self.find_overlap_nodes()
        self.generate_aligned_datasets(args.align_dir)

    def build_net(self, data_path, reader_cls):
        network = reader_cls()
        network.read_network(data_path)
        network.print_summary()
        return network

    def detect_single_instance(self):
        node1_name = 'handshake'
        node2_name = 'deal'
        rel = self.cn_net.graph.find_rel_by_node_name(node1_name, node2_name)
        if rel is not None:
            for x in rel:
                print(x)
        rel = self.cn_net.graph.find_rel_by_node_name(node2_name, node1_name)
        if rel is not None:
            for x in rel:
                print(x)

    def find_overlap_edges(self):
        overlap_edges_set =set()
        overlap_edges_list = list()
        overlap_edges_nodes = set()
        overlap_rels = set()
        newly_add_triples=set()
        
        
        overlap_edges_num=0
        count = list() 
        def add_edges(edge, direction="src->tgt"):
            #triple_old = (edge.relation.name, edge.src.name, edge.tgt.name)
            if direction=="src->tgt":
                src = edge.src.name
                tgt = edge.tgt.name
            elif direction=="tgt->src":
                src = edge.tgt.name
                tgt = edge.src.name

            rel_cn = self.cn_net.graph.find_rel_by_node_name(src, tgt)
            if rel_cn is not None:
                for rel in rel_cn:
                    triple = (rel, src, tgt)
                    overlap_edges_list.append(triple)
                    overlap_edges_set.add(triple)
                    overlap_edges_nodes.update([src, tgt])
                    overlap_rels.add(rel)

                    #replace the "FW-REL" with retrieved ConceptNet relation
                    triple_old = (self.sw_net.rel_token, src, tgt)
                    if triple_old in self.sw_net.edge_set: 
                        self.sw_net.edge_set.discard(triple_old)
                    else:
                        newly_add_triples.add(triple)
                    self.sw_net.edge_set.add(triple)
                    count.append(1) 

        for i, edge in enumerate(self.sw_net.graph.iter_edges()):
            add_edges(edge, "src->tgt")
            add_edges(edge, "tgt->src")
        
        overlap_edges_num= len(overlap_edges_set)
        print("overlap_eges_num: {} with {} nodes ".format(overlap_edges_num, len(overlap_edges_nodes)))
        print("count: {}, overlap_edges: {}".format(sum(count), len(overlap_edges_set)))
        print("Newly add {} triples to SWOW, total :{}".format(\
            len(self.sw_net.edge_set) - self.sw_net.graph.edgeCount, len(self.sw_net.edge_set)))
        #for x in newly_add_triples:
        #    print(x)

        assert overlap_edges_set.issubset(self.cn_net.edge_set)
        assert len(self.sw_net.edge_set.intersection(self.cn_net.edge_set))== len(overlap_edges_set), "{} {}".format( len(self.sw_net.edge_set.intersection(self.cn_net.edge_set)), len(overlap_edges_set))
    
        return overlap_edges_set, overlap_edges_nodes, overlap_rels
    
    def find_overlap_nodes(self):
        sw_nodes = self.sw_net.graph.node2id.keys()
        cn_nodes = self.cn_net.graph.node2id.keys()
        overlap_nodes = set(sw_nodes).intersection(set(cn_nodes))
        print("overlap_nodes num : {}".format(len(overlap_nodes)))
        return overlap_nodes

    def sample_test_valid(self, overlap_edges_set, overlap_edges_nodes, sample_node_size=3000):
        '''sample overlap nodes and overlap triples'''
        sampled_nodes = list(random.sample(overlap_edges_nodes, sample_node_size))
        random.shuffle(sampled_nodes)
        random.shuffle(list(overlap_edges_set))

        sample_nodes_test_pool = sampled_nodes[:int(sample_node_size/2)]
        sample_nodes_valid_pool = sampled_nodes[int(sample_node_size/2):]

        sampled_edges_test = set()
        sampled_edges_valid = set()
        sampled_nodes_test = set()
        sampled_nodes_valid = set()

        for triple in overlap_edges_set:
            rel, src, tgt = triple
            if src in sample_nodes_valid_pool and tgt in sample_nodes_valid_pool:
                sampled_edges_valid.add(triple)
                sampled_nodes_valid.update([src, tgt])
            elif src in sample_nodes_test_pool and tgt in sample_nodes_test_pool:
                sampled_edges_test.add(triple)
                sampled_nodes_test.update([src, tgt])

        assert sampled_nodes_valid.isdisjoint(sampled_nodes_test)
        assert sampled_edges_valid.isdisjoint(sampled_edges_test)

        assert sampled_edges_valid.issubset(self.cn_net.edge_set)
        assert sampled_edges_test.issubset(self.cn_net.edge_set)

        test_net = Triple2GraphReader(sampled_edges_test, 'test')
        valid_net = Triple2GraphReader(sampled_edges_valid, 'valid')

        return test_net, valid_net

    def generate_aligned_datasets(self, out_dir):
        gap_triples = 1e4
        gap_nodes = 1e4
        sample_try=0
        while gap_triples>50 or gap_nodes>10:
            sample_try+=1
            print("Try {} times sampling...".format(sample_try))
            test_net, valid_net  =self.sample_test_valid(self.overlap_edges_set, self.overlap_edges_nodes)
            gap_triples = abs(len(test_net.edge_set) - len(valid_net.edge_set))
            gap_nodes = abs(len(test_net.graph.node2id.keys()) - len(valid_net.graph.node2id.keys()))

        self.plot_sampled_graph_statistics(test_net, valid_net)
        write_relation_triples(self.overlap_edges_set, out_dir+"/rel_triples_overlap12", inp_order='rht')

        def remove_sampled_from_train(net):
            self.overlap_nodes -=net.graph.node2id.keys()
            self.overlap_edges_set -= net.edge_set

            self.cn_net.edge_set -=net.edge_set
            self.sw_net.edge_set -=net.edge_set
        
        cn_edges_num_ori = len(self.cn_net.edge_set)
        sw_edges_num_ori = len(self.sw_net.edge_set)

        remove_sampled_from_train(test_net)
        remove_sampled_from_train(valid_net)

        def check_data(net, edges_num_ori):
            edges_num_cur = len(net.edge_set) + len(test_net.edge_set) + len(valid_net.edge_set)
            assert edges_num_cur == edges_num_ori, "current: {}, ori: {}".format(edges_num_cur, edges_num_ori)
        check_data(self.cn_net, cn_edges_num_ori)
        check_data(self.sw_net, sw_edges_num_ori)

        print("Finish sampling .... ")
        write_eval_to_files(out_dir, test_net, 'test')
        write_eval_to_files(out_dir, valid_net, 'valid')
        write_train_to_files(out_dir, self.cn_net, self.sw_net, self.overlap_edges_set,\
                                self.overlap_nodes, self.overlap_rels)
    
    def plot_sampled_graph_statistics(self, test_net, valid_net):

        def sampled_graph_statistics(sampled_net, columns):
            degree_list= list()
            for node_name in sampled_net.graph.node2id.keys():
                degree = sampled_net.graph.nodes[sampled_net.graph.node2id[node_name]].get_degree()
                cn_degree = self.cn_net.graph.nodes[self.cn_net.graph.node2id[node_name]].get_degree()
                sw_degree = self.sw_net.graph.nodes[self.sw_net.graph.node2id[node_name]].get_degree()
                degree_list.append([node_name, degree, cn_degree, sw_degree])

            df = pd.DataFrame(degree_list, columns=columns)
            return df

        
        test_df = sampled_graph_statistics(test_net, columns=['node_test', 'test_degree', 'cn_degree', 'sw_degree'])
        valid_df = sampled_graph_statistics(valid_net, columns=['node_valid', 'valid_degree', 'cn_degree', 'sw_degree'])

        f,  ax = plt.subplots(3, 2, figsize=(10, 10), sharey='row', sharex='row')

        sns.distplot(test_df["test_degree"], hist=True, kde=False, norm_hist=False, rug=False, label="test_degree", ax=ax[0,0], color='g')
        sns.distplot(test_df["cn_degree"], hist=True, kde=False, norm_hist=False, rug=False, label="cn_degree", ax=ax[1,0], color='r')
        sns.distplot(test_df["sw_degree"], hist=True, kde=False, norm_hist=False, rug=False, label="sw_degree", ax=ax[2,0], color='b')

        sns.distplot(valid_df["valid_degree"], hist=True, kde=False, norm_hist=False, rug=False, label="valid_degree", ax=ax[0,1], color='g' )
        sns.distplot(valid_df["cn_degree"], hist=True, kde=False, norm_hist=False, rug=False, label="cn_degree", ax=ax[1,1], color='r')
        sns.distplot(valid_df["sw_degree"], hist=True, kde=False, norm_hist=False, rug=False, ax=ax[2,1], color='b')
        #ax.set_ylabel(ylabel)
        #ax.set_xlabel("Degree")
        #ax.lines[1].set_linestyle("--")
        plt.legend()
        #plt.supertitle("Ent degrees in various graphs.", fontsize=14)
        plt.show()
        
        plt.savefig('log/{}.png'.format("ent_degree_distribution"), format='png')

    def compare_old_version(self, new_triples):
        old_triples = set()
        with open('./data/swow/conceptnet_swow_edges.overlap') as f:
            data = f.readlines()

        for inst in data:
            inst = inst.strip()
            if inst:
                inst = inst.split('\t')
                rel, src, tgt = inst
                weight = 1.0
                src = src.lower()
                tgt = tgt.lower()
                old_triples.add((rel, tgt, src))

        diff = old_triples.difference(new_triples)
        print("old - new, total: {}".format(len(diff)))
        #for x in diff:
        #    print("old ", x)

        diff = new_triples.difference(old_triples)
        print("new - old total: {}".format(len(diff)))
        #for x in diff:
        #    print("new ", x)

if __name__=='__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--conceptnet_source_file',type=str, default='./data/cn100k/cn100k_train_valid_test.txt')
    parser.add_argument('--swow_source_file',type=str, default='./data/swow/swow_triple_freq2.filter')

    parser.add_argument('--overlap_nodes_file',type=str, default="./data/swow/conceptnet_swow_vocab.overlap")
    parser.add_argument('--lemma_vocab',action='store_true')
    parser.add_argument('--align_dir',type=str, default="data/alignment/C_S_V0.1")
    parser.add_argument('--overwrite_swow_triple', action='store_true')
    parser.add_argument('--swow_cn_rel_links', type=str, default=None)

    args= parser.parse_args()

    Alignment(args)

    #print("detecting case")
    #node1_name = 'paper'
    #node2_name = 'write'

    #rel = cn_net.graph.find_rel_by_node_name(node1_name, node2_name)
    #if rel is not None:
    #    for x in rel:
    #        print(x)

    #node1_name = 'although'
    #node2_name = 'but'
    #rel = cn_net.graph.find_rel_by_node_name(node1_name, node2_name)
    #if rel is not None:
    #    for x in rel:
    #        print(x)
    #else:
    #    print("rel doesn't exist")

