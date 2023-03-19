
import scipy.sparse as sp
import numpy as np


def encode_onehot(labels):
    """
    将标签转换为one-hot编码形式
    :param labels: 标签--list
    :return: 标签的one-hot形式
    """
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i,:] for i,c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="dataset/cora/", dataset="cora"):
    """加载cora数据"""
    print('load_data...')
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
    return features.todense(), adj, labels


def normalize_adj(adj, symmetric=True):
    """
    对邻接矩阵进行归一化处理
    :param adj: 邻接矩阵(密集矩阵)
    :param symmetric: 是否对称
    :return: 归一化后的邻接矩阵
    """
    # 如果邻接矩阵为对称矩阵，得到对称归一化邻接矩阵D^(-1/2) * A * D^(-1/2)
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(axis=1)), -1/2).flatten(), 0)
        a_norm = d.dot(adj).dot(d).tocsr()
    # 如果邻接矩阵不是对称矩阵，得到随机游走正则化拉普拉斯算子D^(-1) * A
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    """
    在邻接矩阵中加入自连接(因为自身信息很重要)
    """
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, nums_sample):
    """
    构造样本掩码
    """
    mask = np.zeros(nums_sample)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    """
    数据集划分
    """
    # 训练集索引列表
    idx_train = range(1700)
    # 验证集索引列表
    idx_val = range(1700, 2000)
    # 测试集索引列表
    idx_test = range(1700, 2700)
    # 训练集样本标签
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    # 验证集样本标签
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_val[idx_val] = y[idx_val]
    # 测试集样本标签
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_test[idx_test] = y[idx_test]
    # 训练数据的样本掩码
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def categorical_crossentropy(preds, labels):
    """
    定义分类交叉熵
    :param preds:模型对样本的输出数组
    :param labels:样本的one-hot标签数组
    :return:样本的平均交叉熵损失
    """
    return np.mean(-np.log(np.extract(labels, preds)))

def accuracy(preds, labels):
    """
    定义准确率函数
    :param preds: 模型对样本的输出数组
    :param labels: 样本的one-hot标签数组
    :return: 样本的平均准确率
    """
    return np.mean(np.equal(np.argmax(labels,axis=1), np.argmax(preds,axis=1)))


def evaluate_preds(preds, labels, indices):
    """
    评估样本划分的损失函数和准确率
    :param preds:对于样本的预测值
    :param labels:样本的标签one-hot向量
    :param indices:样本的索引集合
    :return:交叉熵损失函数列表、准确率列表
    """
    split_loss = list()
    split_acc = list()
    for y_split, idx_split in zip(labels, indices):
        # 计算每一个样本划分的交叉熵损失函数
        split_loss.append(categorical_crossentropy(preds[idx_split],y_split[idx_split]))
        # 计算每一个样本划分的准确率
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))
    return split_loss, split_acc