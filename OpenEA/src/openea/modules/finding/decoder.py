import tensorflow as tf

class ConvTransE:
    def __init__(self, num_entities, num_relations, embedding_dim, input_dropout, hideden_dropout, feature_map_dropout, out_channels):
        """
        Difference from ConvE: no reshaping after stacking e_1 and e_r
        """
        
        self.embedding_dim = embedding_dim
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.feature_map_dropout = feature_map_dropout
        self.filter = 
        self.stride = 1

        self.kernel_size = 5
        self.in_channels = 2
        self.out_channels = out_channels 
        self.filter = tf.zeros([self.kernel_size, self.in_channels, self.out_channels])
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.cur_ent_embeds = None
        self.cur_rel_embeds = None

    def calc_score(self, training):

        ent_embedding = self.cur_embeds
        rel_embedding = self.rel_embeds
        batch_size = e1.shape[0]

        self.pos_e1 = tf.placeholder(tf.int32, shape=[None], name="e1")
        self.pos_rel = tf.placeholder(tf.int32, shape=[None], name="rel")
        self.pos_target = tf.placeholder(tf.int32, shape=[None], name="rel")

        e1 = tf.expand_dims(self.pos_e1, 1)
        rel = tf.expand_dims(self.pos_rel, 1)
        e1_embedded = ent_embedding[e1]
        rel_embedded = rel_embedding[rel]

        stacked_inputs = tf.concat([e1_embedded, rel_embedded], axis=1)
        stacked_inputs = tf.keras.layers.BatchNormalization(stacked_inputs, training=training)

        x = tf.keras.layers.Dropout(rate=self.input_dropout)(stacked_inputs, training=training)
        x = tf.keras.layers.Conv1D( self.out_channels, 
                                    self.kernel_size, 
                                    padding='same', 
                                    data_format='channels_first')(x)
        x = tf.keras.layers.BatchNormalization(x, training=training)
        x = tf.nn.relu(x)
        x = tf.keras.layers.Dropout(rate=self.feature_map_dropout)(x, training=training)
        x = tf.reshape(x, shape=[batch_size, -1])
        x = tf.keras.layers.Dense(  units= self.embedding_dim, 
                                    activation='linear', 
                                    use_bias=True,
                                    kernel_initializer='glorot_uniform')(x)
        x = tf.keras.layers.Dropout(rate=self.hidden_dropout)(x, training=training)
        x = tf.keras.layers.BatchNormalization(x, training=training)
        x = tf.nn.relu(x)

        x = tf.matmul(x, ent_embedding, transpose_b=True)

        self.pred = tf.nn.sigmoid(x)
        self.conv_transe_loss = self.loss(self.pos_target, self.pred)
        if target is None:
            return self.pred
        else:
            return self.conv_transe_loss

class ConvTransE(nn.Module):
    def __init__(self, num_entities, num_relations, args):

        """
        Difference from ConvE: no reshaping after stacking e_1 and e_r
        """

        super(ConvTransE, self).__init__()

        bert_dims = 1024

        self.no_cuda = args.no_cuda
        if args.bert_concat or args.tying:
            emb_dim = args.embedding_dim + bert_dims
        elif args.bert_mlp:
            emb_dim = 600
        elif args.bert_only or args.bert_sum:
            emb_dim = bert_dims
        else:
            emb_dim = args.embedding_dim

        #if args.swow_concat:
            #emb_dim = emb_dim + args.embedding_dim
        #self.swow_sum = args.swow_sum
        self.beta = 0.5

        if args.gcn_type == "MultiHeadGATLayer":
            num_heads = 8
            emb_dim = args.embedding_dim * num_heads + bert_dims

        self.embedding_dim = emb_dim

        self.w_relation = torch.nn.Embedding(num_relations, emb_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.hidden_drop = torch.nn.Dropout(args.dropout)
        self.feature_map_drop = torch.nn.Dropout(args.feature_map_dropout)

        kernel_size = 5
        self.channels = 200

        self.conv1 = nn.Conv1d(2, self.channels, kernel_size, stride=1, padding= int(math.floor(kernel_size/2)))
        # kernel size is odd, then padding = math.floor(kernel_size/2)

        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(self.channels)
        self.bn2 = torch.nn.BatchNorm1d(emb_dim)
        self.fc = torch.nn.Linear(self.channels * emb_dim, emb_dim)
        self.loss = torch.nn.BCELoss()
        self.cur_embedding = None

    def init(self):
        xavier_normal_(self.w_relation.weight.data)

    def forward(self, e1, rel, target):

        embedding = self.cur_embedding
        if not self.no_cuda:
            embedding = embedding.to(torch.cuda.current_device())

        batch_size = e1.shape[0]

        e1 = e1.unsqueeze(1)
        rel = rel.unsqueeze(1)

        e1_embedded = embedding[e1]
        rel_embedded = self.w_relation(rel)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)

        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = tf.keras.layers.Dropout(rate=self.dropout)(x,training=training)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, embedding.t())

        pred = torch.sigmoid(x)
        if target is None:
            return pred
        else:
            return self.loss(pred, target)