
# @inproceedings{feng2020grand,
#   title={Graph Random Neural Network for Semi-Supervised Learning on Graphs},
#   author={Feng, Wenzheng and Zhang, Jie and Dong, Yuxiao and Han, Yu and Luan, Huanbo and Xu, Qian and Yang, Qiang and Kharlamov, Evgeny and Tang, Jie},
#   booktitle={NeurIPS'20},
#   year={2020}
# }

/*
author: Ruimeng Shao
Project: DS-440 Capstone Project
*/








from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
from pygcn.models import GCN, MLP
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--input_droprate', type=float, default=0.5,
                    help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_droprate', type=float, default=0.5,
                    help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--dropnode_rate', type=float, default=0.5,
                    help='Dropnode rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--order', type=int, default=5, help='Propagation step')
parser.add_argument('--sample', type=int, default=4, help='Sampling times of dropnode')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')
parser.add_argument('--dataset', type=str, default='cora', help='Data set')
parser.add_argument('--cuda_device', type=int, default=4, help='Cuda device')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')

#parser.add_argument('--gracedataset', type=str, default='cora')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--config', type=str, default='config.yaml')
#dataset = 'citeseer'
#dataset = 'pubmed'
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
# torch.cuda.set_device(args.cuda_device)
dataset = args.dataset
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
A, features, labels, idx_train, idx_val, idx_test = load_data(dataset)
idx_unlabel = torch.range(idx_train.shape[0], labels.shape[0]-1, dtype=int)

# MLP Model and optimizer
modelmlp = MLP(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            input_droprate=args.input_droprate,
            hidden_droprate=args.hidden_droprate,
            use_bn = args.use_bn)
mlpoptimizer = optim.Adam(modelmlp.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# set parameters
config = yaml.load(open(args.config), Loader=SafeLoader)[dataset]
learning_rate = config['learning_rate']
num_hidden = config['num_hidden']
num_proj_hidden = config['num_proj_hidden']
activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
base_model = ({'GCNConv': GCNConv})[config['base_model']]
num_layers = config['num_layers']

drop_edge_rate_1 = config['drop_edge_rate_1']
drop_edge_rate_2 = config['drop_edge_rate_2']
drop_feature_rate_1 = config['drop_feature_rate_1']
drop_feature_rate_2 = config['drop_feature_rate_2']
tau = config['tau']
num_epochs = config['num_epochs']
weight_decay = config['weight_decay']



def get_dataset(path, name):
        assert name in ['cora', 'CiteSeer', 'PubMed', 'DBLP']
        name = 'dblp' if name == 'DBLP' else name

        return (CitationFull if name == 'dblp' else Planetoid)(
            path,
            name,
            transform = T.NormalizeFeatures())

path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
dataset = get_dataset(path, args.dataset)
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)


# GCN Model, encoder and optimizer
encoder = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)



if args.cuda:
    modelmlp.cuda()
    features = features.cuda()
    A = A.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx_unlabel = idx_unlabel.cuda()

def propagate(feature, A, order):
    #feature = F.dropout(feature, args.dropout, training=training)
    x = feature
    y = feature
    for i in range(order):
        x = torch.spmm(A, x).detach_()
        #print(y.add_(x))
        y.add_(x)
        
    return y.div_(order+1.0).detach_()

def rand_prop(features, training):
    n = features.shape[0]
    drop_rate = args.dropnode_rate
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
    
    if training:
            
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)

        # features = masks.cuda() * features
        features = masks * features
            
    else:
            
        features = features * (1. - drop_rate)
    features = propagate(features, A, args.order)    
    return features

def consis_loss(logps, temp=args.tem):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)
    
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return args.lam * loss

# def contrastive_train(model: Model, x, edge_index, feature):
#     model.train()
#     optimizer.zero_grad()
#     edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
#     edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
#     x_1 = drop_feature(x, drop_feature_rate_1)
#     # x_2 = drop_feature(x, drop_feature_rate_2)
#     # x_1 = rand_prop(x, training=True)
#     x_2 = feature
#     z1 = model(x_1, edge_index_1)
#     z2 = model(x_2, edge_index_2)

#     loss = model.loss(z1, z2, batch_size=0)
#     loss.backward()
#     optimizer.step()

#     return loss.item()

def contrastive_train(model: Model, x, edge_index, z2):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    x = drop_feature(x, drop_feature_rate_2)
    # x_2 = drop_feature(x, drop_feature_rate_2)
    # x_1 = rand_prop(x, training=True)
    z1 = model(x, edge_index_1)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def train(epoch, model: Model, edge_index):
    t = time.time()
    
    X = drop_feature(features, drop_feature_rate_1)
    modelmlp.train()
    mlpoptimizer.zero_grad()
    X_list = []
    K = args.sample

    grace_loss_test = []

    for k in range(K):
        X_list.append(rand_prop(X, training=True))
        # feature2 = drop_feature(X, drop_feature_rate_1)
        # X_list.append(feature2)
        

    
    # K = K * 2 -1

    output_list = []
    for k in range(K):
        output_list.append(torch.log_softmax(modelmlp(X_list[k]), dim=-1))


    

    # grace_loss = contrastive_train(model, X, data.edge_index, feature)

    loss_train = 0.

    for k in range(K):
        #contrastive loss betwen MLP and GCN 
        grace_loss_test = contrastive_train(model, X_list[k], data.edge_index, modelmlp(X_list[k]))
        loss_train = loss_train + F.nll_loss(output_list[k][idx_train], labels[idx_train]) + grace_loss_test
    
    loss_train = loss_train/K
    

    loss_consis = consis_loss(output_list)
    loss_train = loss_train + loss_consis

    acc_train = accuracy(output_list[0][idx_train], labels[idx_train])
    loss_train.backward()
    mlpoptimizer.step()

    if not args.fastmode:
        modelmlp.eval()
        X = rand_prop(X,training=False)
        output = modelmlp(X)
        output = torch.log_softmax(output, dim=-1)
        
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.item(), acc_val.item()
def Train():
    # Train model
    t_total = time.time()
    loss_values = []
    acc_values = []
    bad_counter = 0
    # best = args.epochs + 1
    loss_best = np.inf
    acc_best = 0.0

    loss_mn = np.inf
    acc_mx = 0.0

    best_epoch = 0

    for epoch in range(args.epochs):
        # if epoch < 200:
        #   l, a = train(epoch, True)
        #   loss_values.append(l)
        #   acc_values.append(a)
        #   continue

        l, a = train(epoch, model, data.edge_index)
        loss_values.append(l)
        acc_values.append(a)

        print(bad_counter)

        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:# or epoch < 400:
            if loss_values[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch

            loss_mn = np.min((loss_values[-1], loss_mn))
            acc_mx = np.max((acc_values[-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1

        # print(bad_counter, loss_mn, acc_mx, loss_best, acc_best, best_epoch)
        if bad_counter == args.patience:
            print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
            print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))





def test():
    modelmlp.eval()
    X = features
    X = rand_prop(X, training=False)
    output = modelmlp(X)
    output = torch.log_softmax(output, dim=-1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
Train()
test()
