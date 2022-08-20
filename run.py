import paddle
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
from paddle.io import Dataset
import pandas as pd
import numpy as np
from visualdl import LogWriter

from models import MFBasedModel
from models import GMFBasedModel
from models import DNNBasedModel


class TensorDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        # y = paddle.reshape(y, [len(y), -1])
        # self.data = paddle.concat([X, y], axis=-1)
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.label[idx]
        return X, y

    def __len__(self):
        return len(self.data)


def padding(array, maxlen):
    for i, line in enumerate(array):
        if len(line) > maxlen:
            array[i] = line[-maxlen:]
        else:
            array[i] = np.pad(line, (0, maxlen - len(line)), 'constant', constant_values=(0, 0))
    return array


class Run:

    def __init__(self, config):
        self.use_cuda = config['use_cuda']
        self.base_model = config['base_model']
        self.root = config['root']
        self.ratio = config['ratio']
        self.task = config['task']
        self.src = config['src_tgt_pairs'][self.task]['src']
        self.tgt = config['src_tgt_pairs'][self.task]['tgt']
        self.uid_all = config['src_tgt_pairs'][self.task]['uid']
        self.iid_all = config['src_tgt_pairs'][self.task]['iid']
        self.batchsize_src = config['src_tgt_pairs'][self.task]['batchsize_src'
        ]
        self.batchsize_tgt = config['src_tgt_pairs'][self.task]['batchsize_tgt'
        ]
        self.batchsize_meta = config['src_tgt_pairs'][self.task][
            'batchsize_meta']
        self.batchsize_map = config['src_tgt_pairs'][self.task]['batchsize_map'
        ]
        self.batchsize_test = config['src_tgt_pairs'][self.task][
            'batchsize_test']
        self.batchsize_aug = self.batchsize_src
        self.epoch = config['epoch']
        self.emb_dim = config['emb_dim']
        self.meta_dim = config['meta_dim']
        self.num_fields = config['num_fields']
        self.lr = config['lr']
        if config['wd'] == 0:
            self.wd = None
        self.input_root = self.root + "_" + str(int(self.ratio[0] * 10)
                                                ) + '_' + str(int(self.ratio[1] * 10)
                                                              ) + '/tgt_' + self.tgt + '_src_' + self.src
        self.src_path = self.input_root + '/train_src.csv'
        self.tgt_path = self.input_root + '/train_tgt.csv'
        self.meta_path = self.input_root + '/train_meta.csv'
        self.test_path = self.input_root + '/test.csv'
        self.results = {'tgt_mae': 10, 'tgt_rmse': 10, 'aug_mae': 10,
                        'aug_rmse': 10, 'emcdr_mae': 10, 'emcdr_rmse': 10,
                        'ptupcdr_mae': 10, 'ptupcdr_rmse': 10}

    def seq_extractor(self, x):
        x = x.rstrip(']').lstrip('[').split(', ')
        for i in range(len(x)):
            try:
                x[i] = int(x[i])
            except:
                x[i] = self.iid_all
        return np.array(x)

    def read_log_data(self, path, batchsize, history=False):
        if not history:
            cols = ['uid', 'iid', 'y']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data = pd.read_csv(path, header=None)
            data.columns = cols
            X = paddle.to_tensor(data[x_col].values, dtype='float32')
            y = paddle.to_tensor(data[y_col].values, dtype='float32')
            # if self.use_cuda:
            #     X = X.cuda()
            #     y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batch_size=batchsize, shuffle=True)
            return data_iter
        else:
            data = pd.read_csv(path, header=None)
            cols = ['uid', 'iid', 'y', 'pos_seq']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data.columns = cols
            pos_seq = np.array(padding(data.pos_seq.map(self.seq_extractor).values, 20).tolist())
            pos_seq = paddle.to_tensor(pos_seq, dtype='float32')
            id_fea = paddle.to_tensor(data[x_col].values, dtype='float32')
            X = paddle.concat([id_fea, pos_seq], axis=1)
            y = paddle.to_tensor(data[y_col].values, dtype='float32')
            # if self.use_cuda:
            #     X = X.cuda()
            #     y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batch_size=batchsize, shuffle=True)
            return data_iter

    def read_map_data(self):
        cols = ['uid', 'iid', 'y', 'pos_seq']
        data = pd.read_csv(self.meta_path, header=None)
        data.columns = cols
        X = paddle.to_tensor(data['uid'].unique(), dtype='float32')
        y = paddle.to_tensor(np.array(range(X.shape[0])), dtype='float32')
        # if self.use_cuda:
        #     X = X.cuda()
        #     y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, batch_size=self.batchsize_map, shuffle=True)
        return data_iter

    def read_aug_data(self):
        cols_train = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        src = pd.read_csv(self.src_path, header=None)
        src.columns = cols_train
        tgt = pd.read_csv(self.tgt_path, header=None)
        tgt.columns = cols_train
        X_src = paddle.to_tensor(src[x_col].values, dtype='float32')
        y_src = paddle.to_tensor(src[y_col].values, dtype='float32')
        X_tgt = paddle.to_tensor(tgt[x_col].values, dtype='float32')
        y_tgt = paddle.to_tensor(tgt[y_col].values, dtype='float32')
        X = paddle.concat([X_src, X_tgt])
        y = paddle.concat([y_src, y_tgt])
        # if self.use_cuda:
        #     X = X.cuda()
        #     y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, batch_size=self.batchsize_aug, shuffle=True)
        return data_iter

    def get_data(self):
        print('========Reading data========')
        data_src = self.read_log_data(self.src_path, self.batchsize_src)
        print('src {} iter / batchsize = {} '.format(len(data_src), self.
                                                     batchsize_src))
        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print('tgt {} iter / batchsize = {} '.format(len(data_tgt), self.
                                                     batchsize_tgt))
        data_meta = self.read_log_data(self.meta_path, self.batchsize_meta,
                                       history=True)
        print('meta {} iter / batchsize = {} '.format(len(data_meta), self.
                                                      batchsize_meta))
        data_map = self.read_map_data()
        print('map {} iter / batchsize = {} '.format(len(data_map), self.
                                                     batchsize_map))
        data_aug = self.read_aug_data()
        print('aug {} iter / batchsize = {} '.format(len(data_aug), self.
                                                     batchsize_aug))
        data_test = self.read_log_data(self.test_path, self.batchsize_test,
                                       history=True)
        print('test {} iter / batchsize = {} '.format(len(data_test), self.
                                                      batchsize_test))
        return data_src, data_tgt, data_meta, data_map, data_aug, data_test

    def get_model(self):
        if self.base_model == 'MF':
            model = MFBasedModel(self.uid_all, self.iid_all, self.
                                 num_fields, self.emb_dim, self.meta_dim)
        elif self.base_model == 'DNN':
            model = DNNBasedModel(self.uid_all, self.iid_all, self.
                                  num_fields, self.emb_dim, self.meta_dim)
        elif self.base_model == 'GMF':
            model = GMFBasedModel(self.uid_all, self.iid_all, self.
                                  num_fields, self.emb_dim, self.meta_dim)
        else:
            raise ValueError('Unknown base model: ' + self.base_model)
        return model

    def get_optimizer(self, model):
        optimizer_src = paddle.optimizer.Adam(parameters=model.src_model.parameters(
        ), learning_rate=self.lr, weight_decay=self.wd)
        optimizer_tgt = paddle.optimizer.Adam(parameters=model.tgt_model.parameters(
        ), learning_rate=self.lr, weight_decay=self.wd)
        optimizer_meta = paddle.optimizer.Adam(parameters=model.meta_net.parameters(
        ), learning_rate=self.lr, weight_decay=self.wd)
        optimizer_aug = paddle.optimizer.Adam(parameters=model.aug_model.parameters(
        ), learning_rate=self.lr, weight_decay=self.wd)
        optimizer_map = paddle.optimizer.Adam(parameters=model.mapping.parameters(),
                                              learning_rate=self.lr, weight_decay=self.wd)
        return (optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug,
                optimizer_map)

    def eval_mae(self, model, data_loader, stage):
        print('Evaluating MAE:')
        # paddle eval() without no_grad
        model.eval()
        targets, predicts = list(), list()
        mse_loss = paddle.nn.MSELoss()
        for batch_id, (X, y) in enumerate(data_loader):
            pred = model(X, stage)
            targets.extend(y.squeeze(1).tolist())
            predicts.extend(pred.tolist())
        targets = paddle.to_tensor(targets)
        predicts = paddle.to_tensor(predicts)
        return F.l1_loss(targets, predicts).item(), paddle.sqrt(mse_loss(targets,
                                                                    predicts)).item()

    def train(self, data_loader, model, criterion, optimizer, epoch, stage,
              mapping=False, log_visual=None):
        print('Training Epoch {}:'.format(epoch + 1))
        model.train()
        for batch_id, data in enumerate(data_loader):
            X = data[0]
            y = data[1]
            optimizer.clear_grad()
            if mapping:
                src_emb, tgt_emb = model(X, stage)
                loss = criterion(src_emb, tgt_emb)
            else:
                pred = model(X, stage)
                loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            if log_visual is not None:
                log_visual.add_scalar(tag="train/loss", step=batch_id, value=loss)

    def update_results(self, mae, rmse, phase):
        if mae < self.results[phase + '_mae']:
            self.results[phase + '_mae'] = mae
        if rmse < self.results[phase + '_rmse']:
            self.results[phase + '_rmse'] = rmse

    def TgtOnly(self, model, data_tgt, data_test, criterion, optimizer):
        print('=========TgtOnly========')
        log_visual = LogWriter("./visualDL_log/TgtOnly")
        for i in range(self.epoch):
            self.train(data_tgt, model, criterion, optimizer, i, stage= \
                'train_tgt', log_visual=log_visual)
            mae, rmse = self.eval_mae(model, data_test, stage='test_tgt')
            self.update_results(mae, rmse, 'tgt')
            log_visual.add_scalar(tag="mae", step=i, value=mae)
            print('MAE: {} RMSE: {}'.format(mae, rmse))

    def DataAug(self, model, data_aug, data_test, criterion, optimizer):
        print('=========DataAug========')
        log_visual = LogWriter("./visualDL_log/DataAug")
        for i in range(self.epoch):
            self.train(data_aug, model, criterion, optimizer, i, stage= \
                'train_aug', log_visual=log_visual)
            mae, rmse = self.eval_mae(model, data_test, stage='test_aug')
            self.update_results(mae, rmse, 'aug')
            log_visual.add_scalar(tag="mae", step=i, value=mae)
            print('MAE: {} RMSE: {}'.format(mae, rmse))

    def CDR(self, model, data_src, data_map, data_meta, data_test,
            criterion, optimizer_src, optimizer_map, optimizer_meta):
        print('=====CDR Pretraining=====')
        log_visual = LogWriter("./visualDL_log/CDR")
        for i in range(self.epoch):
            self.train(data_src, model, criterion, optimizer_src, i, stage= \
                'train_src', log_visual=log_visual)
        print('==========EMCDR==========')
        log_visual = LogWriter("./visualDL_log/EMCDR")
        for i in range(self.epoch):
            self.train(data_map, model, criterion, optimizer_map, i, stage= \
                'train_map', mapping=True, log_visual=log_visual)
            mae, rmse = self.eval_mae(model, data_test, stage='test_map')
            self.update_results(mae, rmse, 'emcdr')
            log_visual.add_scalar(tag="mae", step=i, value=mae)
            print('MAE: {} RMSE: {}'.format(mae, rmse))
        print('==========PTUPCDR==========')
        log_visual = LogWriter("./visualDL_log/PTUPCDR")
        for i in range(self.epoch):
            self.train(data_meta, model, criterion, optimizer_meta, i,
                       stage='train_meta', log_visual=log_visual)
            mae, rmse = self.eval_mae(model, data_test, stage='test_meta')
            self.update_results(mae, rmse, 'ptupcdr')
            log_visual.add_scalar(tag="mae", step=i, value=mae)
            print('MAE: {} RMSE: {}'.format(mae, rmse))

    def main(self):
        model = self.get_model()
        data_src, data_tgt, data_meta, data_map, data_aug, data_test = (self
                                                                        .get_data())
        (optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug,
         optimizer_map) = self.get_optimizer(model)
        criterion = paddle.nn.MSELoss()
        self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)
        self.DataAug(model, data_aug, data_test, criterion, optimizer_aug)
        self.CDR(model, data_src, data_map, data_meta, data_test, criterion,
                 optimizer_src, optimizer_map, optimizer_meta)
        paddle.save(model.state_dict(), "checkpoint/last.pdparams")
        print(self.results)
