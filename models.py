import paddle


class LookupEmbedding(paddle.nn.Layer):

    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = paddle.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = paddle.nn.Embedding(iid_all + 1, emb_dim)

    def forward(self, x):
        x = x.cast('int64')
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
        emb = paddle.concat([uid_emb, iid_emb], axis=1)
        return emb


class MetaNet(paddle.nn.Layer):

    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        self.event_K = paddle.nn.Sequential(paddle.nn.Linear(emb_dim,
            emb_dim), paddle.nn.ReLU(), paddle.nn.Linear(emb_dim, 1,
            bias_attr=False))
        self.event_softmax = paddle.nn.Softmax(axis=1)
        self.decoder = paddle.nn.Sequential(paddle.nn.Linear(emb_dim,
            meta_dim), paddle.nn.ReLU(), paddle.nn.Linear(meta_dim, emb_dim *
            emb_dim))

    def forward(self, emb_fea, seq_index):
        mask = (seq_index == 0).cast("float32")
        event_K = self.event_K(emb_fea)
        t = event_K - paddle.unsqueeze(mask, 2) * 100000000.0
        att = self.event_softmax(t)
        his_fea = paddle.sum(att * emb_fea, 1)
        output = self.decoder(his_fea)
        return output.squeeze(1)


class GMFBase(paddle.nn.Layer):

    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = paddle.nn.Linear(emb_dim, 1, bias_attr=False)

    def forward(self, x):
        emb = self.embedding.forward(x)
        x = emb[:, 0, :] * emb[:, 1, :]
        x = self.linear(x)
        return x.squeeze(1)


class DNNBase(paddle.nn.Layer):

    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = paddle.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        emb = self.embedding.forward(x)
        x = paddle.sum(self.linear(emb[:, 0, :]) * emb[:, 1, :], 1)
        return x


class MFBasedModel(paddle.nn.Layer):

    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim_0):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.tgt_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.aug_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim_0)
        self.mapping = paddle.nn.Linear(emb_dim, emb_dim, bias_attr=False)

    def forward(self, x, stage):
        if stage == 'train_src':
            emb = self.src_model.forward(x)
            x = paddle.sum(emb[:, 0, :] * emb[:, 1, :], axis=1)
            return x
        elif stage in ['train_tgt', 'test_tgt']:
            emb = self.tgt_model.forward(x)
            x = paddle.sum(emb[:, 0, :] * emb[:, 1, :], axis=1)
            return x
        elif stage in ['train_aug', 'test_aug']:
            emb = self.aug_model.forward(x)
            x = paddle.sum(emb[:, 0, :] * emb[:, 1, :], axis=1)
            return x
        elif stage in ['train_meta', 'test_meta']:
            x = x.cast('int64')
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))
            ufea = self.src_model.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).reshape([-1, self.emb_dim, self.emb_dim])
            uid_emb = paddle.bmm(uid_emb_src, mapping)
            emb = paddle.concat([uid_emb, iid_emb], 1)
            output = paddle.sum(emb[:, 0, :] * emb[:, 1, :], axis=1)
            return output
        elif stage == 'train_map':
            x = x.cast('int64')
            src_emb = self.src_model.uid_embedding(x.unsqueeze(1)).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.uid_embedding(x.unsqueeze(1)).squeeze()
            return src_emb, tgt_emb
        elif stage == 'test_map':
            x = x.cast('int64')
            uid_emb = self.mapping.forward(self.src_model.uid_embedding(x[:,
                0].unsqueeze(1)).squeeze())
            emb = self.tgt_model.forward(x)
            emb[:, 0, :] = uid_emb
            x = paddle.sum(emb[:, 0, :] * emb[:, 1, :], axis=1)
            return x


class GMFBasedModel(paddle.nn.Layer):

    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = GMFBase(uid_all, iid_all, emb_dim)
        self.tgt_model = GMFBase(uid_all, iid_all, emb_dim)
        self.aug_model = GMFBase(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim)
        self.mapping = paddle.nn.Linear(emb_dim, emb_dim, bias_attr=False)

    def forward(self, x, stage):
        if stage == 'train_src':
            x = self.src_model.forward(x)
            return x
        elif stage in ['train_tgt', 'test_tgt']:
            x = self.tgt_model.forward(x)
            return x
        elif stage in ['train_aug', 'test_aug']:
            x = self.aug_model.forward(x)
            return x
        elif stage in ['test_meta', 'train_meta']:
            x = x.cast('int64')
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].
                unsqueeze(1))
            uid_emb_src = self.src_model.embedding.uid_embedding(x[:, 0].
                unsqueeze(1))
            ufea = self.src_model.embedding.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).reshape([-1, self.emb_dim, self.emb_dim])
            uid_emb = paddle.bmm(uid_emb_src, mapping)
            emb = paddle.concat([uid_emb, iid_emb], 1)
            output = self.tgt_model.linear(emb[:, 0, :] * emb[:, 1, :])
            return output.squeeze(1)
        elif stage == 'train_map':
            x = x.cast('int64')
            src_emb = self.src_model.embedding.uid_embedding(x.unsqueeze(1)
                ).squeeze()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.embedding.uid_embedding(x.unsqueeze(1)
                ).squeeze()
            return src_emb, tgt_emb
        elif stage == 'test_map':
            x = x.cast('int64')
            uid_emb = self.mapping.forward(self.src_model.embedding.
                uid_embedding(x[:, 0].unsqueeze(1)))
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].
                unsqueeze(1))
            emb = paddle.concat([uid_emb, iid_emb], 1)
            x = self.tgt_model.linear(emb[:, 0, :] * emb[:, 1, :])
            return x.squeeze(1)


class DNNBasedModel(paddle.nn.Layer):

    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = DNNBase(uid_all, iid_all, emb_dim)
        self.tgt_model = DNNBase(uid_all, iid_all, emb_dim)
        self.aug_model = DNNBase(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim)
        self.mapping = paddle.nn.Linear(emb_dim, emb_dim, bias_attr=False)

    def forward(self, x, stage):
        if stage == 'train_src':
            x = self.src_model.forward(x)
            return x
        elif stage in ['train_tgt', 'test_tgt']:
            x = self.tgt_model.forward(x)
            return x
        elif stage in ['train_aug', 'test_aug']:
            x = self.aug_model.forward(x)
            return x
        elif stage in ['test_meta', 'train_meta']:
            x = x.cast('int64')
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].
                unsqueeze(1))
            uid_emb_src = self.src_model.linear(self.src_model.embedding.
                uid_embedding(x[:, 0].unsqueeze(1)))
            ufea = self.src_model.embedding.iid_embedding(x[:, 2:])
            mapping = self.meta_net.forward(ufea, x[:, 2:]).reshape([-1, self.emb_dim, self.emb_dim])
            uid_emb = paddle.bmm(uid_emb_src, mapping)
            emb = paddle.concat([uid_emb, iid_emb], 1)
            output = paddle.sum(emb[:, 0, :] * emb[:, 1, :], 1)
            return output
        elif stage == 'train_map':
            x = x.cast('int64')
            src_emb = self.src_model.linear(self.src_model.embedding.
                uid_embedding(x.unsqueeze(1)).squeeze())
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.linear(self.tgt_model.embedding.
                uid_embedding(x.unsqueeze(1)).squeeze())
            return src_emb, tgt_emb
        elif stage == 'test_map':
            x = x.cast('int64')
            uid_emb = self.mapping.forward(self.src_model.linear(self.
                src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))))
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].
                unsqueeze(1))
            emb = paddle.concat([uid_emb, iid_emb], 1)
            x = paddle.sum(emb[:, 0, :] * emb[:, 1, :], 1)
            return x
