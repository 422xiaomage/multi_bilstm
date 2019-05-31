import tensorflow as tf

class Bilstm(object):
    def __init__(self, vocab_size,hidden_dims,embedding=None, embedding_size=300, hidden_size=128, max_length=256,
                 dropoutKeepProb=0.5,numClass=3):
        self.embedding = embedding
        self.embedding_size = embedding_size
        # 参数用于设置多层双向lstm，参数是列表结构，列表每多一个元素，就多一层双向lstm。而不是增加单向的lstm层数
        self.hidden_dims = hidden_dims
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.dropoutKeepProb = dropoutKeepProb
        self.vocab_size = vocab_size
        self.numClass = numClass


        # 词嵌入层
        self.input_X = tf.placeholder(tf.int32, [None, self.max_length], name="input_X")
        self.input_Y = tf.placeholder(tf.int32, [None, 1], name="input_Y")
        with tf.name_scope("embedding"):
            if self.embedding is not None:
                # 这里将词向量设置成了变量，所以在训练过程中，也会对原始词向量进行微调，如果不需要，
                # 可以将这一句修改为注释掉的代码段
                self.W = tf.Variable(tf.cast(self.embedding, dtype=tf.float32,name="word2vec"), name="W")
                # self.W = tf.constant(tf.cast(self.embedding, dtype=tf.float32,name="word2vec"), name="W")
            else:
                self.W = tf.Variable(tf.truncated_normal([self.vocab_size,self.embedding_size],stddev=1,
                                                         dtype=tf.float32), name="W")
                # self.W = tf.constant(tf.truncated_normal([self.vocab_size,self.embedding_size],stddev=1,
                # dtype=tf.float32), name="W")
            # 词序号的向量化操作
            self.embeddingwords = tf.nn.embedding_lookup(self.W, self.input_X)
            # 对标签进行onne_hot编码,这里要注意self.input_Y的数据类型必须时int型
            self.input_Y_one_hot = tf.cast(tf.one_hot(self.input_Y, self.numClass, name="Y_onehot"), dtype=tf.float32)
        # 卷积层和池化层
        with tf.name_scope("bilstm"):
            for index, hidden_dim in enumerate(self.hidden_dims):
                with tf.name_scope("bilstm"+str(index)):
                    # tf.nn.rnn_cell.MultiRNNCell()的使用注意，每一层得重新定义，然后放在一个列表里，否则会报错
                    # fw_cells = tf.nn.rnn_cell.MultiRNNCell([fw_cell]*3, state_is_tuple=True)这种写法会爆维度错误
                    fw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size,
                                                                                         state_is_tuple=True),
                                                            output_keep_prob=self.dropoutKeepProb)
                    fw_cell1 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size,
                                                                                          state_is_tuple=True),
                                                             output_keep_prob=self.dropoutKeepProb)
                    fw_cells = tf.nn.rnn_cell.MultiRNNCell([fw_cell,fw_cell1], state_is_tuple=True)
                    bw_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size,
                                                                                         state_is_tuple=True),
                                                            output_keep_prob=self.dropoutKeepProb)
                    bw_cell1 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size,
                                                                                          state_is_tuple=True),
                                                             output_keep_prob=self.dropoutKeepProb)
                    bw_cells = tf.nn.rnn_cell.MultiRNNCell([bw_cell,bw_cell1])
                    outputs, current_state = tf.nn.bidirectional_dynamic_rnn(fw_cells,bw_cells,self.embeddingwords,
                                                                             dtype=tf.float32,
                                                                             scope="bi-lstm" + str(index))
                    self.embeddingwords = tf.concat(outputs, 2)
            finalOutput = self.embeddingwords[:, -1, :]
            flat_length = self.hidden_size * 2
            self.finalOutput = tf.reshape(finalOutput, [-1, flat_length])

        # 定义全连接层
        with tf.name_scope("hidden_"):
            output_W = tf.get_variable(
                "output_W",
                shape=[flat_length, self.hidden_size],
                initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.15, shape=[self.hidden_size]), name="output_b")
        out = tf.nn.xw_plus_b(self.finalOutput, output_W, output_b, name="ceng1")
        # 定一输出层
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[hidden_size, self.numClass],
                initializer=tf.contrib.layers.xavier_initializer())
            output_bb = tf.Variable(tf.constant(0.1, shape=[self.numClass]), name="b")
            self.predictions = tf.nn.xw_plus_b(out, W, output_bb, name="predictions")

            self.output = tf.cast(tf.arg_max(self.predictions, 1), tf.float32, name="category")
        # 计算三元交叉熵损失
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predictions,
                                                                               labels=self.input_Y_one_hot))
        # 优化器
