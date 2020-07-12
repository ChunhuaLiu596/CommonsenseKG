import math
import tensorflow as tf
import time
import sys,os
import random
import numpy as np
import pandas as pd
import multiprocessing as mp
from tensorflow.contrib.layers import l2_regularizer

from openea.modules.utils.util import task_divide
from openea.models.basic_model import BasicModel
from openea.modules.utils.util import load_session
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.optimizers import generate_optimizer
import openea.modules.train.batch as bat
from openea.modules.load.kgs import KGs
from openea.modules.bootstrapping.alignment_finder import find_potential_alignment_greedily, check_new_alignment
from openea.modules.finding.evaluation_completion import calculate_rank_bidirection, write_rank_to_file
from openea.modules.base.losses import positive_loss
from sklearn.metrics import confusion_matrix
from openea.modules.finding.evaluation import valid, test, early_stop
from openea.modules.utils.confusion_matrix_pretty_print import  pretty_plot_confusion_matrix, plot_confusion_matrix_from_data
import openea.modules.load.read as rd
np.set_printoptions(threshold=sys.maxsize)

def generate_batch(kgs: KGs,  batch_size, step, neg_triple_num, train_kg):
    if train_kg=="kg1":
        pos_triples, neg_triples = bat.generate_relation_triple_batch(kgs.kg1.relation_triples_list, None,
                                                                    kgs.kg1.relation_triples_set, None,
                                                                    kgs.kg1.entities_list, None,
                                                                    batch_size, step,
                                                                    None, None, neg_triple_num)

    elif train_kg=="kg2":
        pos_triples, neg_triples = bat.generate_relation_triple_batch(None, kgs.kg2.relation_triples_list,
                                                                    None, kgs.kg2.relation_triples_set,
                                                                    None, kgs.kg2.entities_list,
                                                                    batch_size, step,
                                                                    None, None, neg_triple_num)
    elif train_kg=="kg12":
        pos_triples, neg_triples = bat.generate_relation_triple_batch(kgs.kg1.relation_triples_list,
                                                                    kgs.kg2.relation_triples_list,
                                                                    kgs.kg1.relation_triples_set,
                                                                    kgs.kg2.relation_triples_set,
                                                                    kgs.kg1.entities_list, kgs.kg2.entities_list,
                                                                    batch_size, step,
                                                                    None, None, neg_triple_num)
    return pos_triples, neg_triples


def generate_batch_queue(kgs: KGs, batch_size,  steps, neg_triple_num, out_queue, train_kg):
    for step in steps:
        pos_triples, neg_triples= generate_batch(kgs, batch_size, step, neg_triple_num, train_kg)
        out_queue.put((pos_triples, neg_triples))


class IPTransE(BasicModel):

    def __init__(self):
        super().__init__()
        self.paths1, self.paths2 = None, None
        self.global_step_triple = tf.Variable(0, trainable=False)
        self.global_step_rel = tf.Variable(0, trainable=False)
        self.kg_eval = None
        self.session = load_session()

    def init(self):
        self.beta1=self.args.beta1
        self.beta2=self.args.beta2
        self.entities_num = self.kgs.entities_num
        self.relations_num = self.kgs.relations_num
        self.relations_mask_num = self.kgs.relations_mask_num

        self._define_placeholders()
        self._define_variables()
        if self.args.predict_relation:
            self._define_rel_graph()
        self._define_embed_graph()
        self._define_eval_graph()
        #self._define_alignment_graph()

        if self.args.mode=="train":
            self._define_summary_writers()
            tf.compat.v1.global_variables_initializer().run(session=self.session)
            self.saver = tf.compat.v1.train.Saver()
            self.checkpoint_dir =  "{}model/".format(self.out_folder)
            self.save_path = self.saver.save(self.session, "{}model.ckpt".format(self.checkpoint_dir))
        #if self.args.load_pretrain_model:
        #    self.reload_model()

        # customize parameters
        #if self.args.train_kg=="kg12":
        #    assert self.args.alignment_module == 'sharing'
        if self.args.train_kg=="kg1" or self.args.train_kg=="kg2":
            assert self.args.alignment_module != 'sharing'
        assert self.args.init == 'normal'
        assert self.args.neg_sampling == 'uniform'
        assert self.args.optimizer == 'Adagrad'
        assert self.args.eval_metric == 'inner'
        assert self.args.loss_norm == 'L2'

        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True

        assert self.args.margin > 0.0
        assert self.args.neg_triple_num == 1
        assert self.args.sim_th > 0.0

    def check_norm(self):
        self.ent_embeds =  tf.nn.l2_normalize(self.ent_embeds, 1) if self.args.ent_l2_norm else self.ent_embeds
        self.rel_embeds =  tf.nn.l2_normalize(self.rel_embeds, 1) if self.args.rel_l2_norm else self.rel_embeds
        #print('checked entity norm and relation norm')

    def _define_variables(self):
        with tf.compat.v1.variable_scope('embeddings'):
            if self.args.load_pretrain_emb:
                ent_embeds, rel_embeds = self.load_pretrain_emb()
                with tf.compat.v1.variable_scope('ent_embeddings'):
                    self.ent_embeds = tf.Variable(ent_embeds, trainable=False, name="ent_embeds")
                with tf.compat.v1.variable_scope('rel_embeddings'):
                    self.rel_embeds = tf.Variable(rel_embeds, trainable=False, name="rel_embeds")
                    self.kg1_rel_embeds = tf.slice(self.rel_embeds, [0,0], [self.relations_mask_num,self.args.dim])
                self.check_norm()
            else:
                with tf.compat.v1.variable_scope('ent_embeddings'):
                    self.ent_embeds = init_embeddings([self.entities_num, self.args.dim], 'ent_embeds',
                                                        self.args.init, self.args.ent_l2_norm)

                with tf.compat.v1.variable_scope('rel_embeddings'):
                    self.rel_embeds = init_embeddings([self.relations_num , self.args.dim], 'rel_embeds',
                                                        self.args.init, self.args.rel_l2_norm)
                    if self.args.predict_relation:
                        self.kg1_rel_embeds = tf.slice(self.rel_embeds, [0,0], [self.kgs.relations_mask_num, self.args.dim])
                        print("self.kg1_rel_embeds", self.kg1_rel_embeds)
                if self.args.mode=="train":
                    self.sum_ent = tf.compat.v1.summary.histogram(name=self.ent_embeds.op.name, values=self.ent_embeds)
                    self.sum_rel = tf.compat.v1.summary.histogram(name=self.rel_embeds.op.name, values=self.rel_embeds)

    def _define_summary_writers(self):
        train_dir = self.out_folder + "train/"
        valid1_dir = self.out_folder + "valid1/"
        valid2_dir = self.out_folder + "valid2/"

        self.writer_train = tf.compat.v1.summary.FileWriter(train_dir, self.session.graph)
        self.writer_valid1 = tf.compat.v1.summary.FileWriter(valid1_dir)
        self.writer_valid2 = tf.compat.v1.summary.FileWriter(valid2_dir)
        print("train summary dir: {}".format(train_dir))
        print("valid1 summary dir: {}".format(valid1_dir))
        print("valid2 summary dir: {}".format(valid2_dir))
        print("Open tensorboard: tensorboard --logdir=run1:\"{}\",run2:\"{}\",run3:\"{}\" ".format(train_dir, valid1_dir, valid2_dir))

    def _define_placeholders(self):
        with tf.name_scope("train_placeholders"):
            self.pos_hs = tf.compat.v1.placeholder(tf.int32, shape=[None], name="pos_hs")
            self.pos_rs = tf.compat.v1.placeholder(tf.int32, shape=[None], name="pos_rs")
            self.pos_ts = tf.compat.v1.placeholder(tf.int32, shape=[None], name="pos_ts")

            self.neg_hs = tf.compat.v1.placeholder(tf.int32, shape=[None], name="neg_hs")
            self.neg_rs = tf.compat.v1.placeholder(tf.int32, shape=[None], name="neg_rs")
            self.neg_ts = tf.compat.v1.placeholder(tf.int32, shape=[None], name="neg_ts")
            self.pos_rs_mask = tf.compat.v1.placeholder(tf.bool, shape=[None], name="pos_rs_mask")

        with tf.name_scope("eval_placeholders"):
            self.eval_phs = tf.compat.v1.placeholder(tf.int32, shape=[None], name="eval_phs")
            self.eval_prs = tf.compat.v1.placeholder(tf.int32, shape=[None], name="eval_prs")
            self.eval_pts = tf.compat.v1.placeholder(tf.int32, shape=[None], name="eval_pts")
            self.eval_prs_mask = tf.compat.v1.placeholder(tf.bool, shape=[None], name="eval_prs_mask")

            self.eval_candidates = tf.compat.v1.placeholder(tf.int32, shape=[None], name="eval_cadidates")

    def _generate_transe_loss(self, phs, prs, pts, nhs, nrs, nts):
        if self.args.loss_norm == "L2":
            pos_score = tf.reduce_sum(tf.pow(phs + prs - pts, 2), 1)
            neg_score = tf.reduce_sum(tf.pow(nhs + nrs - nts, 2), 1)
        else:
            pos_score = tf.reduce_sum(tf.abs(phs + prs - pts), 1)
            neg_score = tf.reduce_sum(tf.abs(nhs + nrs - nts), 1)
        return tf.reduce_sum(tf.maximum(pos_score + self.args.margin - neg_score, 0))

    def _generate_rel_cross_entropy_loss(self, preds, labels):
        """Softmax cross-entropy loss for relations."""
        loss = - tf.reduce_sum(labels*tf.log(preds + 1e-9),1)
        return tf.reduce_sum(loss, name="cross_entropy_loss")
        #loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        #return tf.reduce_sum(loss, name="cross_entropy_loss")

    def _generate_rel_entropy_loss(self, preds):
        '''preds: normalized logits'''
        loss = - tf.reduce_sum(preds*tf.log(preds + 1e-9),1)
        return tf.reduce_sum(loss, name="entropy_loss")

    def _generate_rel_acc(self, preds, labels):
        correct = tf.equal(tf.argmax(preds,1),tf.argmax(labels,1))
        acc = tf.reduce_mean(tf.cast(correct, "float"), name="accuracy")
        return acc

    def _define_rel_emb(self, hs, ts, inp_dropout=0, hid_dropout=0, training=False):
        with tf.compat.v1.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            with tf.compat.v1.variable_scope("train_rel_types", reuse=tf.AUTO_REUSE):

                inp = tf.concat([hs, ts], -1)
                if training:
                    inp = tf.nn.dropout(inp, keep_prob=1-inp_dropout)
                hidden1 = tf.layers.dense(inputs=inp, units= self.args.rel_hidden_dim,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                        kernel_regularizer='l2',
                                        reuse=tf.AUTO_REUSE, name='w_rel_hidden1')

                if training:
                    hidden1 = tf.nn.dropout(hidden1, keep_prob=1-hid_dropout)

                hidden2 = tf.layers.dense(inputs=hidden1, units= self.args.rel_hidden_dim,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                        kernel_regularizer='l2',
                                        reuse=tf.AUTO_REUSE, name='w_rel_hidden2')

                logits = tf.layers.dense(inputs=hidden2, units=self.kgs.relations_mask_num,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                        kernel_regularizer='l2',
                                        reuse=tf.AUTO_REUSE, name="w_rel_output")

                logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)
                logits_normalized = tf.nn.softmax(logits, name='rel_logits')
                rs = tf.matmul(logits_normalized, self.kg1_rel_embeds) #(b, 34) (34, d)
        return  rs, logits_normalized

    def _define_pos_rs(self, phs, pts, pos_rs, pos_rs_mask, inp_dropout=0, hid_dropout=0, training=False):
        '''pos_rs_pres is the unnormalized relation type distribution'''
        prs_pred, pos_rs_pred = self._define_rel_emb(phs, pts, inp_dropout, hid_dropout, training)
        pos_rs_pred_cn = tf.boolean_mask(pos_rs_pred, pos_rs_mask)
        pos_rs_label_cn = tf.boolean_mask(pos_rs, pos_rs_mask)
        pos_rs_label_cn = tf.one_hot(pos_rs_label_cn, self.kgs.relations_mask_num)
        return  prs_pred, pos_rs_pred_cn, pos_rs_label_cn, pos_rs_pred

    def _define_embed_graph(self):
        with tf.name_scope("train-triple"):
            phs = tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs)
            pts = tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts)
            nhs = tf.nn.embedding_lookup(self.ent_embeds, self.neg_hs)
            nts = tf.nn.embedding_lookup(self.ent_embeds, self.neg_ts)

            if not self.args.predict_relation:
                prs = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
                #nrs = tf.nn.embedding_lookup(self.rel_embeds, self.neg_rs)
            else:
                prs, _, _, _ = self._define_pos_rs(phs, pts, self.pos_rs, self.pos_rs_mask, self.args.rel_inp_dropout, self.args.rel_hid_dropout, True)

            nrs = prs
            self.triple_loss= self._generate_transe_loss(phs, prs, pts, nhs, nrs, nts)
            tf.compat.v1.summary.scalar("triple_loss", self.triple_loss)
            self._add_summary(training=True)

            if self.args.mode=="train":
                self.var_list_emb = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="embeddings")
                for op in self.var_list_emb:
                    print("updating: {}".format(op.name))

                learning_rate = tf.compat.v1.train.exponential_decay(self.args.learning_rate, self.global_step_triple,
                                                                    10000, 0.96, staircase=True)
                self.optimizer = generate_optimizer(self.triple_loss, learning_rate, var_list=self.var_list_emb, opt=self.args.optimizer)

    def _define_rel_graph(self):
        with tf.name_scope("train-rel"):
            phs = tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs)
            pts = tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts)
            prs = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
            nhs = tf.nn.embedding_lookup(self.ent_embeds, self.neg_hs)
            nts = tf.nn.embedding_lookup(self.ent_embeds, self.neg_ts)

            _, pos_rs_pred_cn, pos_rs_label_cn, pos_rs_pred = self._define_pos_rs(phs, pts, self.pos_rs, self.pos_rs_mask, self.args.rel_inp_dropout, self.args.rel_hid_dropout, True)

            self.rel_cross_entropy_loss = self.beta1 * self._generate_rel_cross_entropy_loss(pos_rs_pred_cn, pos_rs_label_cn)
            self.rel_entropy_loss = self.beta2 * self._generate_rel_entropy_loss(pos_rs_pred)

            self.train_loss_rel = self.rel_cross_entropy_loss +  self.rel_entropy_loss

            self.rel_acc = self._generate_rel_acc(pos_rs_pred_cn, pos_rs_label_cn)

            self.pos_rs_pred_cn = tf.argmax(pos_rs_pred_cn, 1)
            self.pos_rs_pred = tf.argmax(pos_rs_pred, 1)

            self.entropy_nan = tf.math.is_nan(self.rel_entropy_loss)
            self.cross_entropy_nan = tf.math.is_nan(self.rel_cross_entropy_loss)
            self.logits = pos_rs_pred

            if self.args.mode=="train":
                self.var_list_rel = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="train_rel_types")
                for op in self.var_list_rel:
                    print("updating: {}".format(op.name))

                learning_rate = tf.compat.v1.train.exponential_decay(self.args.learning_rate, self.global_step_rel,
                                                                    5000, 0.96, staircase=True)
                self.rel_optimizer = generate_optimizer(self.train_loss_rel, learning_rate, var_list=self.var_list_rel, opt=self.args.optimizer)

    def _add_summary(self, training=True, metrics=None):
        if training:
            train_sum_lt = tf.compat.v1.summary.scalar("triple_loss", self.triple_loss)
            if self.args.predict_relation:
                train_sum_ra = tf.compat.v1.summary.scalar("rel_acc", self.rel_acc)
                train_sum_lp = tf.compat.v1.summary.scalar("rel_entropy_loss", self.rel_entropy_loss)
                train_sum_lr =tf.compat.v1.summary.scalar("rel_cross_entropy_loss", self.rel_cross_entropy_loss)

                self.train_merged = tf.compat.v1.summary.merge([train_sum_lt, train_sum_ra,train_sum_lr, train_sum_lp])
            else:
                self.train_merged = tf.compat.v1.summary.merge([train_sum_lt])
        else:
            if self.args.predict_relation:
                valid_sum_ra =tf.compat.v1.summary.scalar("rel_acc", self.eval_rel_acc)
                valid_sum_lp = tf.compat.v1.summary.scalar("rel_entropy_loss", self.eval_rel_entropy_loss)
                valid_sum_lr =tf.compat.v1.summary.scalar("rel_cross_entropy_loss", self.eval_rel_cross_entropy_loss)
                #valid_sum_l =tf.compat.v1.summary.scalar("eval_loss", self.eval_loss)
                if self.args.mode=="train":
                    self.valid_merged = tf.compat.v1.summary.merge([valid_sum_ra, valid_sum_lr, valid_sum_lp, self.sum_rel])
                                                    #self.sum_ent, self.sum_rel])
    def _add_metrics_summary(self, metrics, writer, epoch):
        for key, value in metrics.items():
                sum_metrics = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='metrics/'+key, simple_value=value), ])
                writer.add_summary(sum_metrics, epoch)

    def _define_eval_graph(self):
        with tf.name_scope('evaluation'):
            phs = tf.nn.embedding_lookup(self.ent_embeds, self.eval_phs)
            prs = tf.nn.embedding_lookup(self.rel_embeds, self.eval_prs)
            pts = tf.nn.embedding_lookup(self.ent_embeds, self.eval_pts)

            # how about the training batch? what's the e2s
            candidates = tf.nn.embedding_lookup(self.ent_embeds, self.eval_candidates)
            candidates = tf.expand_dims(candidates, 0)
            #candidates = tf.expand_dims(self.ent_embeds, 0)
            if self.args.predict_relation:
                prs_predicted = self._define_eval_relation_types(phs, prs, pts, self.eval_prs, self.eval_prs_mask)
                self._define_eval_triple_loss(phs, prs_predicted, pts)
                self.eval_loss = self.eval_triple_loss + self.eval_rel_cross_entropy_loss + self.eval_rel_entropy_loss
            else:
                self._define_eval_triple_loss(phs, prs, pts)

            self._define_eval_completion(phs, prs, pts, candidates)
            self._add_summary(training=False)

    def _define_eval_completion(self, phs, prs, pts, candidates):
        with tf.name_scope('link'):
            if self.args.loss_norm == "L2":
                self.distance_t_pred = tf.reduce_sum(tf.pow(tf.expand_dims(phs + prs, 1) - candidates, 2), 2)

                self.distance_h_pred = tf.reduce_sum(tf.pow(candidates + tf.expand_dims(prs - pts, 1), 2), 2)
                #prts = tf.expand_dims(prs - phs, 1)
                #self.distance_h_pred = tf.reduce_sum(tf.pow(candidates + prts, 2), 2)

    def _define_eval_relation_types(self, phs, prs, pts, prs_label, prs_mask, type='valid'):
        '''evaluate relation accuracy for positive triples'''
        #assert self.args.predict_relation
        start = time.time()
        prs, prs_pred_cn, prs_label_cn, prs_pred = self._define_pos_rs(phs, pts, prs_label, prs_mask)

        self.eval_rel_acc = self._generate_rel_acc(prs_pred_cn, prs_label_cn)
        with tf.name_scope("eval_rel_losses"):
            self.eval_rel_cross_entropy_loss = self.beta1 * self._generate_rel_cross_entropy_loss(prs_pred_cn, prs_label_cn)
            self.eval_rel_entropy_loss = self.args.beta2 * self._generate_rel_entropy_loss(prs_pred)

            #self.relation_pred = tf.argmax(tf.nn.softmax(prs_pred_cn), 1)
            #self.relation_pred = tf.argmax(prs_pred_cn, 1)
            #self.relation_label  = tf.argmax(prs_label_cn, 1)
            self.relation_pred = prs_pred_cn
            self.relation_label  = prs_label_cn
        return prs

    def _define_eval_triple_loss(self, phs, prs, pts):
        with tf.name_scope("eval_rel_losses"):
            if self.args.loss_norm == "L2":
                self.eval_triple_loss = positive_loss(phs, prs, pts, 'L2')
            else:
                self.eval_triple_loss = positive_loss(phs, prs, pts, 'L1')


    def launch_triple_training_1epo(self, epoch, triple_steps, steps_tasks, batch_queue):
        start = time.time()
        for steps_task in steps_tasks:
            mp.Process(target=generate_batch_queue,
                            args=(self.kgs, self.args.batch_size,
                            steps_task, self.args.neg_triple_num, batch_queue,
                            self.args.train_kg)).start()
        fetches = {"triple_loss": self.triple_loss,
                    "train_op": self.optimizer }

        if self.args.mode=="train":
            fetches.update({"summary": self.train_merged})

        if self.args.predict_relation:
            fetches.update({"rel_acc": self.rel_acc,
                            "rel_cross_entropy_loss": self.rel_cross_entropy_loss,
                            "rel_entropy_loss": self.rel_entropy_loss,
                            "pos_rs_pred": self.pos_rs_pred,
                            "pos_rs_pred_cn": self.pos_rs_pred_cn,
                            "entropy_nan": self.entropy_nan,
                            "cross_entropy_nan": self.cross_entropy_nan,
                            "logits": self.logits,
                            })

        epoch_vals = list()
        for step in range(triple_steps):
            pos_triples, neg_triples= batch_queue.get()

            pos_rs=[x[1] for x in pos_triples]
            neg_rs=[x[1] for x in neg_triples]
            assert pos_rs==neg_rs, "pos_rs != neg_rs"

            feed_dict = {self.pos_rs: pos_rs,
                            self.neg_rs: neg_rs,
                            self.pos_hs: [x[0] for x in pos_triples],
                            self.pos_ts: [x[2] for x in pos_triples],
                            self.neg_hs: [x[0] for x in neg_triples],
                            self.neg_ts: [x[2] for x in neg_triples],
                            }
            if self.args.predict_relation:
                pos_rs_mask = [True if r_id in self.kgs.rel_ids_mask else False for r_id in pos_rs]
                feed_dict.update({ self.pos_rs_mask: pos_rs_mask })
            vals = self.session.run(fetches=fetches, feed_dict=feed_dict)
            epoch_vals.append(vals)
            #self.detect_nan(vals)
        self.writer_train.add_summary(vals["summary"], epoch)
        self.writer_train.flush()
        self._print_epoch_vals(epoch_vals, epoch, triple_steps, start)

        if self.kgs.kg1 is not None:
            random.shuffle(self.kgs.kg1.relation_triples_list)
        if self.kgs.kg2 is not None:
            random.shuffle(self.kgs.kg2.relation_triples_list)

    def detect_nan(self, vals):
        if vals["entropy_nan"] or vals["cross_entropy_nan"]:
            print("entropy_nan:{}, cross_entropy_nan:{} relation types causes nan...".format(vals["entropy_nan"], vals["cross_entropy_nan"]))
            print("logits: {}".format(vals["logits"]))
            print(vals["pos_rs_pred"])

    def launch_rel_training_1epo(self, epoch, triple_steps, steps_tasks, batch_queue):
        start = time.time()
        for steps_task in steps_tasks:
            mp.Process(target=generate_batch_queue,
                            args=(self.kgs, self.args.batch_size,
                            steps_task, self.args.neg_triple_num, batch_queue,
                            self.args.train_kg)).start()
        fetches = {"loss": self.train_loss_rel,
                    "train_op": self.rel_optimizer,
                    "summary": self.train_merged}
        if self.args.predict_relation:
            fetches.update({"rel_acc": self.rel_acc,
                            "triple_loss": self.triple_loss,
                            "rel_cross_entropy_loss": self.rel_cross_entropy_loss,
                            "rel_entropy_loss": self.rel_entropy_loss,
                            "pos_rs_pred": self.pos_rs_pred,
                            "pos_rs_pred_cn": self.pos_rs_pred_cn,
                            "entropy_nan": self.entropy_nan,
                            "cross_entropy_nan": self.cross_entropy_nan,
                            "logits": self.logits,
                            })

        epoch_vals = list()
        for step in range(triple_steps):
            pos_triples, neg_triples= batch_queue.get()

            pos_rs=[x[1] for x in pos_triples]
            neg_rs=[x[1] for x in neg_triples]
            assert pos_rs==neg_rs, "pos_rs != neg_rs"

            feed_dict = {self.pos_rs: pos_rs,
                            self.neg_rs: neg_rs,
                            self.pos_hs: [x[0] for x in pos_triples],
                            self.pos_ts: [x[2] for x in pos_triples],
                            self.neg_hs: [x[0] for x in neg_triples],
                            self.neg_ts: [x[2] for x in neg_triples],
                            }
            if self.args.predict_relation:
                pos_rs_mask = [True if r_id < self.kgs.relations_mask_num else False for r_id in pos_rs]
                feed_dict.update({ self.pos_rs_mask: pos_rs_mask })
            vals = self.session.run(fetches=fetches, feed_dict=feed_dict)
            epoch_vals.append(vals)
            #self.detect_nan(vals)
        self.writer_train.add_summary(vals["summary"], epoch)
        self.writer_train.flush()
        #self._print_epoch_vals(epoch_vals, epoch, triple_steps, start)
        if self.kgs.kg1 is not None:
            random.shuffle(self.kgs.kg1.relation_triples_list)
        if self.kgs.kg2 is not None:
            random.shuffle(self.kgs.kg2.relation_triples_list)
        self.check_norm()

    def _print_epoch_vals(self, epoch_vals, epoch, triple_steps, start):
        triple_loss = 0
        rel_acc, rel_cross_entropy_loss, rel_entropy_loss = 0, 0, 0
        for i in range(triple_steps):
            triple_loss += epoch_vals[i]["triple_loss"]
            if self.args.predict_relation:
                rel_acc += epoch_vals[i]["rel_acc"]
                rel_entropy_loss += epoch_vals[i]["rel_entropy_loss"]
                rel_cross_entropy_loss += epoch_vals[i]["rel_cross_entropy_loss"]

        triple_loss /= self.args.batch_size

        if not self.args.predict_relation:
            print('epoch {}, avg. triple_loss: {:.4f}, cost time: {:.4f}s'.format(epoch, triple_loss, time.time() - start))
        else:
            rel_acc /=triple_steps # already compute mean for each pos_batch
            rel_entropy_loss /=self.args.batch_size
            rel_cross_entropy_loss /=self.args.batch_size

            print('epoch {}, avg. triple_loss: {:.4f}, rel_cross_entropy_loss: {:.4f}, rel_entropy_loss: {:.4f}, rel_acc: {:.4f}, cost time: {:.4f}s'.format(epoch, triple_loss, rel_cross_entropy_loss, rel_entropy_loss ,rel_acc, time.time() - start))

    def launch_ptranse_evaluation(self, kg_eval, writer, type='valid', epoch=None, draw_confm=False, save=False):

        if self.args.train_kg=="kg12":
            flag1 = self.launch_ptranse_evaluate_completion(kg_eval, writer, type, epoch, draw_confm, save)
            flag2 = self.launch_ptranse_evaluate_alignment(kg_eval, writer, type, epoch, save)
            return 0.5*(flag1 + flag2)
        else:
            flag1 = self.launch_ptranse_evaluate_completion(kg_eval, writer, type, epoch, draw_confm, save)
            return flag1

    def launch_ptranse_evaluate_alignment(self, kg_eval, writer, type='valid', epoch=None, save=False):
        if type=='valid':
            mr, mrr, hits, _, _ = self.valid(self.args.stop_metric)
        elif type=='test':
            mr, mrr, hits, _, _ = self.test(save)

        if epoch is not None:
            with tf.name_scope("alignment_metrics"):
                metrics = {"align_mr": mr, "align_mrr": mrr,
                            "align_hits1": hits[0], "align_hits10": hits[2]}
                self._add_metrics_summary(metrics, writer, epoch)

        #if type=='valid':
        flag = hits[0] if self.args.stop_metric == 'hits1' else mrr
        return flag

    def launch_ptranse_evaluate_completion(self, kg_eval, writer=None, type='valid', epoch=None, draw_confm=False, save=False):
        triples_num = kg_eval.relation_triples_num
        entities_num = kg_eval.entities_num
        start = time.time()

        fetches={ "triple_loss": self.eval_triple_loss,
                    "distance_t_pred":self.distance_t_pred,
                    "distance_h_pred":self.distance_h_pred}

        if self.args.mode=="train":
            fetches.update({"summary": self.valid_merged})

        if self.args.predict_relation:
            fetches.update({"rel_acc": self.eval_rel_acc,
                    "rel_cross_entropy_loss": self.eval_rel_cross_entropy_loss,
                    "rel_entropy_loss": self.eval_rel_entropy_loss,
                    "relation_pred":self.relation_pred,
                    "relation_label":self.relation_label,
                    "eval_loss": self.eval_loss,
                })

        pos_triples = kg_eval.relation_triples_list
        pos_hs = [x[0] for x in pos_triples]
        pos_rs = [x[1] for x in pos_triples]
        pos_ts = [x[2] for x in pos_triples]
        pos_rs_mask = [True if r_id in self.kgs.rel_ids_mask else False for r_id in pos_rs]

        feed_dict = { self.eval_phs: pos_hs,
                    self.eval_prs: pos_rs,
                    self.eval_pts: pos_ts,
                    self.eval_prs_mask: pos_rs_mask,
                    self.eval_candidates: kg_eval.entities_list,
                    }
        vals = self.session.run(fetches=fetches, feed_dict=feed_dict)
        loss =0
        if self.args.predict_relation:
            rel_acc = vals["rel_acc"]

        mr, mrr, hits, hits_12_list, hits_21_list = calculate_rank_bidirection(kg_eval.local_relation_triples_list, vals["distance_t_pred"], vals["distance_h_pred"], self.args.top_k, kg_eval.local_hr_to_multi_t, kg_eval.local_tr_to_multi_h)

        if epoch is not None:
            writer.add_summary(vals["summary"], epoch)
            with tf.name_scope("completion_metrics"):
                metrics = {"completion_mr": mr, "completion_mrr": mrr,
                        "completion_hits1": hits[0], "completion_hits10": hits[2] }
                self._add_metrics_summary(metrics, writer, epoch)
            writer.flush()


        if self.args.predict_relation:
            print("relation results: valid_rel_acc = {:.4f}, triples: {} ".format(rel_acc, triples_num))
            if draw_confm:
                rel_ids = sorted(set( np.argmax(vals["relation_pred"], 1)) | set(np.argmax(vals["relation_label"], 1)))
                rel_classes = kg_eval.get_rel_classes(rel_ids, self.kgs.relations_classes)
                plot_confusion_matrix_from_data(np.argmax(vals["relation_label"],1), np.argmax(vals["relation_pred"],1), columns= rel_classes,fz=14, figsize=[len(rel_classes)+1, len(rel_classes)+1], show_null_values=0, save_name= self.out_folder+'test_confusion_matrix.png')

        print("completion results: hits@{} = {}, mr = {:.4f}, mrr = {:.4f}, rank_candidates: {}, time = {:.3f} s. ".
                format(self.args.top_k, hits, mr, mrr, entities_num, time.time() - start))

        if save:
            write_rank_to_file(kg_eval, hits_12_list, hits_21_list, kg_eval.local_id_entities_dict,
                                out_folder=self.out_folder, suffix=type)
            if self.args.predict_relation:
                rd.write_rel_score_to_file(vals["relation_label"], vals["relation_pred"], self.out_folder)

        #if type=='valid':
        flag = hits[0] if self.args.stop_metric == 'hits1' else mrr
        return flag

    def run(self):
        t = time.time()
        if self.args.train_kg=="kg1":
            triples_num = self.kgs.kg1.relation_triples_num
        elif self.args.train_kg=="kg2":
            triples_num = self.kgs.kg2.relation_triples_num
        else:
            triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num

        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()

        for epoch in range(1, self.args.max_epoch):
            if self.args.predict_relation:
                if epoch < self.args.max_epoch_rel:
                    self.launch_rel_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue)
                elif epoch % self.args.train_rel_freq ==0:
                    self.launch_rel_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue)

            self.launch_triple_training_1epo(epoch, triple_steps, steps_tasks, training_batch_queue)

            if epoch % self.args.eval_freq == 0:
                print("Evalute on kg1_valid")
                flag_kg1 = self.launch_ptranse_evaluation(self.kgs.kg1_valid, writer=self.writer_valid1, type='valid',epoch=epoch)

                print("\nEvalute on kg2_valid")
                flag_kg2 = self.launch_ptranse_evaluation(self.kgs.kg2_valid, writer=self.writer_valid2, type='valid',epoch=epoch)

                flag = 0.5*(flag_kg1 + flag_kg2)

                self._add_metrics_summary({"stop_metric_"+self.args.stop_metric:flag}, self.writer_valid1, epoch)
                #self.launch_ptranse_evaluation(self.kgs.kg_test,  writer=self.writer_test, type='test', epoch=epoch)

                if flag> self.best_flag:
                    self.best_epoch = epoch
                    self.best_flag = flag
                    print("Model saved in path: {}, best_epoch: {}".format(self.save_path, self.best_epoch))

                if self.args.stop_metric=="max_epoch" and epoch==self.args.max_epoch:
                    break
                elif (epoch - self.best_epoch)% self.args.eval_freq > self.args.early_stop_patience and epoch> self.args.start_valid:
                    break

        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))

        ##### Test #####
        print("Best epoch: {}".format(self.best_epoch))
        print("\nStart testing ...")
        self.reload_model(self.checkpoint_dir)
        print("Test kg1_test")
        self.launch_ptranse_evaluation(self.kgs.kg1_test,  writer=None, type='test', epoch=None, draw_confm=True, save=True)

        print("Test kg2_test")
        self.launch_ptranse_evaluation(self.kgs.kg2_test,  writer=None, type='test', epoch=None, draw_confm=True, save=True)


    def retest(self, save=True):
        self.check_norm()
        if self.args.train_kg=="kg12":
            self.retest_alignment(save)
        self.launch_ptranse_evaluate_completion(self.kgs.kg1_test, writer=None, type='test', epoch=None, draw_confm=False, save=True)
        self.launch_ptranse_evaluate_completion(self.kgs.kg2_test, writer=None, type='test', epoch=None, draw_confm=False, save=True)
