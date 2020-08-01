import math
import multiprocessing as mp
import random
import time
import gc

import tensorflow as tf
import numpy as np
import os

import openea.modules.load.read as rd
import openea.modules.train.batch as bat
from openea.modules.finding.evaluation import early_stop, rank_alignment_bidirection, rank_alignment
from openea.modules.utils.util import generate_out_folder
from openea.modules.utils.util import load_session
from openea.modules.utils.util import task_divide
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.losses import get_loss_func
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.base.mapping import add_mapping_variables, add_mapping_module

from openea.modules.finding.alignment import stable_alignment


class BasicModel:

    def set_kgs(self, kgs):
        self.kgs = kgs

    def set_args(self, args):
        self.args = args
        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division,
                                            self.__class__.__name__)

    def init(self):
        # need to be overwrite
        pass

    def __init__(self):

        self.out_folder = None
        self.args = None
        self.kgs = None

        self.session = None

        self.seed_entities1 = None
        self.seed_entities2 = None
        self.neg_ts = None
        self.neg_rs = None
        self.neg_hs = None
        self.pos_ts = None
        self.pos_rs = None
        self.pos_hs = None
        self.pos_rs_mask = None
        self.eval_pts = None
        self.eval_prs = None
        self.eval_phs = None
        self.eval_prs_mask = None

        self.rel_embeds = None
        self.ent_embeds = None
        self.mapping_mat = None
        self.eye_mat = None

        self.triple_optimizer = None
        self.triple_loss = None
        self.mapping_optimizer = None
        self.mapping_loss = None

        self.mapping_mat = None

        self.best_flag = 0 
        self.best_epoch = 0
        self.flag1 = -1
        self.flag2 = -1
        self.early_stop = False

    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds',
                                                self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                                self.args.init, self.args.rel_l2_norm)

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.neg_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('triple_lookup'):
            phs = tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs)
            prs = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
            pts = tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts)
            nhs = tf.nn.embedding_lookup(self.ent_embeds, self.neg_hs)
            nrs = tf.nn.embedding_lookup(self.rel_embeds, self.neg_rs)
            nts = tf.nn.embedding_lookup(self.ent_embeds, self.neg_ts)
        with tf.name_scope('triple_loss'):
            self.triple_loss = get_loss_func(phs, prs, pts, nhs, nrs, nts, self.args)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                        opt=self.args.optimizer)

    def _define_mapping_variables(self):
        add_mapping_variables(self)

    def _define_mapping_graph(self):
        '''
        initialize mapping weight and generate self.mapping_loss
        '''
        add_mapping_module(self)

    def _eval_valid_embeddings(self):
        if len(self.kgs.valid_links) > 0:
            embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.valid_entities1).eval(session=self.session)
            embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.valid_entities2).eval(session=self.session)
        else:
            embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities1).eval(session=self.session)
            embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities2).eval(session=self.session)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def _eval_test_embeddings(self):
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.test_entities2).eval(session=self.session)
        mapping = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        return embeds1, embeds2, mapping

    def _eval_test_relation_types(self, kg_test):
        self.launch_ptranse_test_relation_types(kg_test)


    def valid_alignment(self, stop_metric):
        embeds1, embeds2, mapping = self._eval_valid_embeddings()
        mr, mrr, hits, hits_12_list, hits_21_list  = rank_alignment_bidirection(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
                            metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        return  mr, mrr, hits, hits_12_list, hits_21_list 

    def test_alignment(self, save=True):
        embeds1, embeds2, mapping = self._eval_test_embeddings()

        mr, mrr, hits, hits_12_list, hits_21_list  = rank_alignment_bidirection(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
                            metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)

        if save:
            rd.save_results(self.out_folder, hits_12_list)

        return  mr, mrr, hits, hits_12_list, hits_21_list 

    def retest_alignment(self,save):
        print(self.__class__.__name__, type(self.__class__.__name__))
        #mapping = None
        #if os.path.exists(new_dir + "mapping_mat.npy"):
        #    print(self.__class__.__name__, "loads mapping mat")
        #    mapping = np.load(new_dir + "mapping_mat.npy")

        embeds1, embeds2, mapping = self._eval_test_embeddings()

        id1_entities_dict = dict()
        for new_id, old_id in enumerate(self.kgs.test_entities1):
            ent = self.kgs.kg1.id_entities_dict.get(old_id)
            id1_entities_dict[new_id] = ent

        id2_entities_dict = dict()
        for new_id, old_id in enumerate(self.kgs.test_entities2):
            ent = self.kgs.kg2.id_entities_dict[old_id]
            id2_entities_dict[new_id] = ent

        print("conventional test:")
        mr_12, mrr_12, hits_12, hits_12_list = rank_alignment(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
                metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        if save:
            hits_12_ent = rd.pairs_id2ent(hits_12_list, id1_entities_dict, id2_entities_dict)
            rd.save_results(self.out_folder, hits_12_ent, 'alignment_results_12')

        print("conventional reversed test:")
        if mapping is not None:
            embeds1 = np.matmul(embeds1, mapping)
            rank_alignment(embeds2, embeds1, None, self.args.top_k, self.args.test_threads_num,
                    metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
        else:
            mr_21, mrr_21, hits_21, hits_21_list = rank_alignment(embeds2, embeds1, mapping, self.args.top_k, self.args.test_threads_num,
                    metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
            if save:
                hits_21_ent = rd.pairs_id2ent(hits_21_list, id2_entities_dict, id2_entities_dict)
                rd.save_results(self.out_folder, hits_21_ent, 'alignment_results_21')

    def save(self):
        ent_embeds = self.ent_embeds.eval(session=self.session)
        rel_embeds = self.rel_embeds.eval(session=self.session)
        mapping_mat = self.mapping_mat.eval(session=self.session) if self.mapping_mat is not None else None
        rd.save_embeddings(self.out_folder, self.kgs, ent_embeds, rel_embeds, None, mapping_mat=mapping_mat)

    def eval_kg1_ent_embeddings(self):
        embeds = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.kg1.entities_list)
        return embeds.eval(session=self.session)

    def eval_kg2_ent_embeddings(self):
        embeds = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.kg2.entities_list)
        return embeds.eval(session=self.session)

    def eval_kg1_useful_ent_embeddings(self):
        embeds = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.useful_entities_list1)
        return embeds.eval(session=self.session)

    def eval_kg2_useful_ent_embeddings(self):
        embeds = tf.nn.embedding_lookup(self.ent_embeds, self.kgs.useful_entities_list2)
        return embeds.eval(session=self.session)

    def launch_training_1epo(self, epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2):
        self.launch_triple_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
        if self.args.alignment_module == 'mapping':
            self.launch_mapping_training_1epo(epoch, triple_steps)

    def launch_triple_training_1epo(self, epoch, triple_steps, steps_tasks, batch_queue, neighbors1, neighbors2):
        start = time.time()
        for steps_task in steps_tasks:
            mp.Process(target=bat.generate_relation_triple_batch_queue,
                       args=(self.kgs.kg1.relation_triples_list, self.kgs.kg2.relation_triples_list,
                             self.kgs.kg1.relation_triples_set, self.kgs.kg2.relation_triples_set,
                             self.kgs.kg1.entities_list, self.kgs.kg2.entities_list,
                             self.args.batch_size, steps_task,
                             batch_queue, neighbors1, neighbors2, self.args.neg_triple_num)).start()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            batch_pos, batch_neg = batch_queue.get()
            batch_loss, _ = self.session.run(fetches=[self.triple_loss, self.triple_optimizer],
                                             feed_dict={self.pos_hs: [x[0] for x in batch_pos],
                                                        self.pos_rs: [x[1] for x in batch_pos],
                                                        self.pos_ts: [x[2] for x in batch_pos],
                                                        self.neg_hs: [x[0] for x in batch_neg],
                                                        self.neg_rs: [x[1] for x in batch_neg],
                                                        self.neg_ts: [x[2] for x in batch_neg]})
            trained_samples_num += len(batch_pos)
            epoch_loss += batch_loss
        epoch_loss /= trained_samples_num
        random.shuffle(self.kgs.kg1.relation_triples_list)
        random.shuffle(self.kgs.kg2.relation_triples_list)
        print('epoch {}, avg. triple loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_mapping_training_1epo(self, epoch, triple_steps):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        for i in range(triple_steps):
            links_batch = random.sample(self.kgs.train_links, len(self.kgs.train_links) // triple_steps)
            batch_loss, _ = self.session.run(fetches=[self.mapping_loss, self.mapping_optimizer],
                                             feed_dict={self.seed_entities1: [x[0] for x in links_batch],
                                                        self.seed_entities2: [x[1] for x in links_batch]})
            epoch_loss += batch_loss
            trained_samples_num += len(links_batch)
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. mapping loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        neighbors1, neighbors2 = None, None
        for i in range(1, self.args.max_epoch + 1):
            self.launch_training_1epo(i, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break
            if self.args.neg_sampling == 'truncated' and i % self.args.truncated_freq == 0:
                t1 = time.time()
                assert 0.0 < self.args.truncated_epsilon < 1.0
                neighbors_num1 = int((1 - self.args.truncated_epsilon) * self.kgs.kg1.entities_num)
                neighbors_num2 = int((1 - self.args.truncated_epsilon) * self.kgs.kg2.entities_num)
                if neighbors1 is not None:
                    del neighbors1, neighbors2
                gc.collect()
                neighbors1 = bat.generate_neighbours(self.eval_kg1_useful_ent_embeddings(),
                                                    self.kgs.useful_entities_list1,
                                                    neighbors_num1, self.args.batch_threads_num)
                neighbors2 = bat.generate_neighbours(self.eval_kg2_useful_ent_embeddings(),
                                                    self.kgs.useful_entities_list2,
                                                    neighbors_num2, self.args.batch_threads_num)
                ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
                print("\ngenerating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))
                gc.collect()
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))

    def reload_model(self, saver=None, checkpoint_dir=None):
        #saver = tf.train.import_meta_graph(self.args.model_meta_path)
        if saver is None:
            saver = tf.compat.v1.train.Saver()
        if checkpoint_dir is None:
            checkpoint_dir = self.args.checkpoint_dir
        saver.restore(self.session, tf.train.latest_checkpoint(checkpoint_dir))

    def load_pretrain_emb(self):
        dir = self.out_folder.split("/")
        new_dir = ""
        for i in range(len(dir) - 2):
            new_dir += (dir[i] + "/")
        exist_file = os.listdir(new_dir)
        #new_dir = new_dir + exist_file[0] + "/"
        ent_embeds = np.loadtxt(new_dir + "ent_embeds.tsv", delimiter="\t", dtype=np.float32)
        rel_embeds = np.loadtxt(new_dir + "rel_embeds.tsv", delimiter="\t", dtype=np.float32)
        return ent_embeds, rel_embeds
