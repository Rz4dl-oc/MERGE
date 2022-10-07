import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.conv import GATConv

import math
import numpy as np

import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph

from TAMDUR import *


class QKVInviBasedAttn(nn.Module):
    def __init__(self, input_size, dk):
        super(QKVInviBasedAttn, self).__init__()
        self.dk = dk
        self.linearQ = nn.Linear(input_size, self.dk).cuda()
        self.linearK = nn.Linear(input_size, self.dk).cuda()

    def forward(self, X, h):
        dk = self.dk
        Q = F.dropout(self.linearQ(h), p=0.5, training=self.training).cuda()   # B*1*Dk
        K = F.dropout(self.linearK(X), p=0.5, training=self.training).cuda()   # B*N*Dk
        A = F.softmax(torch.matmul(Q, K.transpose(-1, -2))/math.sqrt(dk), dim=2).cuda()          # B*1*N
        V = X   # B*N*Dh
        out = torch.matmul(A, V)   # B*1*Dh

        return out


class MERGE_LSTM(nn.Module):
    def __init__(self, nF0, nF, nclass, dropout, nheads, static_features, attn_dk):
        super(MERGE_LSTM, self).__init__()
        self.dropout = dropout
        self.nF0 = nF0
        self.nF = nF
        self.nheads = nheads

        nF2 = nF + static_features
        self.nF2 = nF2

        self.eb = nn.Linear(nF0, nF)

        self.rnn = nn.LSTM(input_size=nF0, hidden_size=nF, num_layers=1, batch_first=True)  # (B, T, F)
        # self.t_attention = LinearSelfAttn(nF).cuda()

        self.gcDy = GATConv(nF2, nF2, num_heads=nheads).cuda()

        self.gc1 = GATConv(nF2, nF2, num_heads=nheads).cuda()

        self.gc2 = GATConv(nF2, nF2, num_heads=nheads).cuda()

        self.gc3_0 = GATConv(nF2, nF2, num_heads=nheads, allow_zero_in_degree=True).cuda()
        self.gc3_1 = GATConv(nF2, nF2, num_heads=nheads, allow_zero_in_degree=True).cuda()

        self.gc4_0 = GATConv(nF2, nF2, num_heads=nheads, allow_zero_in_degree=True).cuda()
        self.gc4_1 = GATConv(nF2, nF2, num_heads=nheads, allow_zero_in_degree=True).cuda()

        self.attention_merger = QKVInviBasedAttn(nF2, attn_dk)

        self.linear = nn.Linear(nF2+nF2, nclass)

    def forward(self, input_tensor, dglGraphs, static_tensor, w):
        npat = input_tensor.shape[0]

        nF2 = self.nF2

        input_tensor = F.dropout(F.relu(self.eb(input_tensor)), p=0.5, training=self.training)

        aft_rnn = F.dropout(self.rnn(input_tensor, None)[0], p=0.5, training=self.training)
        aft_rnn_last = aft_rnn[:, -1, :].cuda()

        A_mat = kneighbors_graph(np.array(aft_rnn_last.detach().cpu().numpy()), w, mode='connectivity',
                                 include_self=True).toarray()

        sp_dy_g = sp.csr_matrix(A_mat)
        dy_dgl_g = dgl.from_scipy(sp_dy_g).to(torch.device("cuda"))

        aft_rnn_last = torch.cat(tensors=(aft_rnn_last, static_tensor), dim=1)

        aft_gnn_dy = F.dropout(F.leaky_relu(self.gcDy(dy_dgl_g, aft_rnn_last)), p=self.dropout, training=self.training)

        aft_gnn1 = F.dropout(F.leaky_relu(self.gc1(dglGraphs[0].to(torch.device("cuda")), aft_rnn_last)),
                             p=self.dropout, training=self.training)
        aft_gnn2 = F.dropout(F.leaky_relu(self.gc2(dglGraphs[1].to(torch.device("cuda")), aft_rnn_last)),
                             p=self.dropout, training=self.training)

        gender_nodes = torch.zeros(size=(int(dglGraphs[2].num_nodes()) - npat, nF2), dtype=torch.float).cuda()
        bft_gnn3 = torch.cat(tensors=(aft_rnn_last, gender_nodes), dim=0)
        aft_gnn3 = F.dropout(F.leaky_relu(self.gc3_0(dglGraphs[2].to(torch.device("cuda")), bft_gnn3)), p=self.dropout,
                             training=self.training)
        aft_gnn3 = F.dropout(F.leaky_relu(self.gc3_1(dglGraphs[3].to(torch.device("cuda")), aft_gnn3)),
                             p=self.dropout, training=self.training)[:npat]

        services_nodes = torch.zeros(size=(int(dglGraphs[4].num_nodes()) - npat, nF2), dtype=torch.float).cuda()
        bft_gnn4 = torch.cat(tensors=(aft_rnn_last, services_nodes), dim=0)
        aft_gnn4 = F.dropout(F.leaky_relu(self.gc4_0(dglGraphs[4].to(torch.device("cuda")), bft_gnn4)), p=self.dropout,
                             training=self.training)
        aft_gnn4 = F.dropout(F.leaky_relu(self.gc4_1(dglGraphs[5].to(torch.device("cuda")), aft_gnn4)),
                             p=self.dropout, training=self.training)[:npat]

        aft_gnn = torch.cat(tensors=(aft_gnn1, aft_gnn2, aft_gnn3, aft_gnn4, aft_gnn_dy), dim=1)

        aft_gnn = self.attention_merger(aft_gnn, aft_rnn_last.unsqueeze(-1).reshape(npat, 1, nF2))

        bft_out = torch.cat(tensors=(aft_rnn_last, aft_gnn.reshape(npat, nF2)), dim=1).cuda()
        out = F.dropout(self.linear(bft_out).cuda(), p=0.5, training=self.training)

        return F.log_softmax(out, dim=1)


# ours TAMDUR
class MERGE_TAMDUR(nn.Module):
    def __init__(self, nF0, nF, nclass, dropout, input_dims, time_steps, nheads, static_features, attn_dk):
        """Dense version of GAT."""
        super(MERGE_TAMDUR, self).__init__()
        self.dropout = dropout
        self.nF0 = nF0
        self.nF = nF
        self.nheads = nheads

        nF2 = nF*len(input_dims) + static_features
        self.nF2 = nF2

        # Some time series method includes embedding layer.
        # self.eb = nn.Linear(nF0, nF)

        self.time_layer = TAMDUR_Layer(total_input_dim=nF0, input_dims=input_dims, hidden_dim=nF, time_steps=time_steps)

        self.gcDy = GATConv(nF2, nF2, num_heads=nheads).cuda()

        self.gc1 = GATConv(nF2, nF2, num_heads=nheads).cuda()

        self.gc2 = GATConv(nF2, nF2, num_heads=nheads).cuda()

        self.gc3_0 = GATConv(nF2, nF2, num_heads=nheads, allow_zero_in_degree=True).cuda()
        self.gc3_1 = GATConv(nF2, nF2, num_heads=nheads, allow_zero_in_degree=True).cuda()

        self.gc4_0 = GATConv(nF2, nF2, num_heads=nheads, allow_zero_in_degree=True).cuda()
        self.gc4_1 = GATConv(nF2, nF2, num_heads=nheads, allow_zero_in_degree=True).cuda()

        self.attention_merger = QKVInviBasedAttn(nF2, attn_dk)

        self.linear = nn.Linear(nF2+nF2, nclass)

    def forward(self, input_tensor, dglGraphs, static_tensor, w, input_ids, delays):
        npat = input_tensor.shape[0]

        nF2 = self.nF2

        # Some time series method includes embedding layer.
        # input_tensor = F.dropout(F.relu(self.eb(input_tensor)), p=0.5, training=self.training)

        aft_rnn = F.dropout(self.time_layer(input_tensor, input_ids, delays), p=0.3, training=self.training)
        aft_rnn_last = aft_rnn.cuda()

        A_mat = kneighbors_graph(np.array(aft_rnn_last.detach().cpu().numpy()), w, mode='connectivity',
                                 include_self=True).toarray()

        sp_dy_g = sp.csr_matrix(A_mat)
        dy_dgl_g = dgl.from_scipy(sp_dy_g).to(torch.device("cuda"))

        aft_rnn_last = torch.cat(tensors=(aft_rnn_last, static_tensor), dim=1)

        aft_gnn_dy = F.dropout(F.leaky_relu(self.gcDy(dy_dgl_g, aft_rnn_last)), p=self.dropout, training=self.training)

        aft_gnn1 = F.dropout(F.leaky_relu(self.gc1(dglGraphs[0].to(torch.device("cuda")), aft_rnn_last)),
                             p=self.dropout, training=self.training)
        aft_gnn2 = F.dropout(F.leaky_relu(self.gc2(dglGraphs[1].to(torch.device("cuda")), aft_rnn_last)),
                             p=self.dropout, training=self.training)

        gender_nodes = torch.zeros(size=(int(dglGraphs[2].num_nodes())-npat, nF2), dtype=torch.float).cuda()
        bft_gnn3 = torch.cat(tensors=(aft_rnn_last, gender_nodes), dim=0)
        aft_gnn3 = F.dropout(F.leaky_relu(self.gc3_0(dglGraphs[2].to(torch.device("cuda")), bft_gnn3)), p=self.dropout,
                             training=self.training)
        aft_gnn3 = F.dropout(F.leaky_relu(self.gc3_1(dglGraphs[3].to(torch.device("cuda")), aft_gnn3)),
                             p=self.dropout, training=self.training)[:npat]

        services_nodes = torch.zeros(size=(int(dglGraphs[4].num_nodes())-npat, nF2), dtype=torch.float).cuda()
        bft_gnn4 = torch.cat(tensors=(aft_rnn_last, services_nodes), dim=0)
        aft_gnn4 = F.dropout(F.leaky_relu(self.gc4_0(dglGraphs[4].to(torch.device("cuda")), bft_gnn4)), p=self.dropout,
                             training=self.training)
        aft_gnn4 = F.dropout(F.leaky_relu(self.gc4_1(dglGraphs[5].to(torch.device("cuda")), aft_gnn4)),
                             p=self.dropout, training=self.training)[:npat]

        aft_gnn = torch.cat(tensors=(aft_gnn1, aft_gnn2, aft_gnn3, aft_gnn4, aft_gnn_dy), dim=1)

        aft_gnn = self.attention_merger(aft_gnn, aft_rnn_last.unsqueeze(-1).reshape(npat, 1, nF2))

        bft_out = torch.cat(tensors=(aft_rnn_last, aft_gnn.reshape(npat, nF2)), dim=1).cuda()
        out = F.dropout(self.linear(bft_out).cuda(), p=0.5, training=self.training)

        return F.log_softmax(out, dim=1)

