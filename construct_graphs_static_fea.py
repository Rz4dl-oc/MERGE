import numpy as np
import torch
import dgl


def construct_graph_numeral_static_feature(prov_tensor, main_tensor, name_prev, theta, low, high):
    tensor = torch.cat(tensors=(prov_tensor, main_tensor), dim=0)
    cluster_size = tensor.shape[0]

    from_l = []
    to_l = []

    for j in range(cluster_size):
        if j % 200 == 0:
            print("finish nodes---", j)
        from_l.append(j)
        to_l.append(j)
        for k in range(j + 1, cluster_size):
            if abs(tensor[j] - tensor[k]) < theta and (tensor[j] <= high and tensor[j] >=low):
                from_l.append(j)
                to_l.append(k)
                from_l.append(k)
                to_l.append(j)
    graph = dgl.graph((torch.tensor(from_l), torch.tensor(to_l)))
    dgl.save_graphs(name_prev + ".dgl", graph)


def construct_graph_category_static_feature(prov_tensor, main_tensor, name_prev):
    tensor = torch.cat(tensors=(prov_tensor, main_tensor), dim=0)
    cluster_size = tensor.shape[0]
    amo_categories = tensor.shape[1]

    from_l0 = []
    to_l0 = []
    from_l1 = []
    to_l1 = []

    for j in range(cluster_size):
        if j % 200 == 0:
            print("finish nodes---", j)

        from_l0.append(j)
        to_l0.append(j)
        from_l1.append(j)
        to_l1.append(j)

        for k in range(amo_categories):
            if tensor[j][k] == 1:
                from_l0.append(j)
                to_l0.append(k + cluster_size)
                from_l1.append(k + cluster_size)
                to_l1.append(j)

    gender_dgl0 = dgl.graph((torch.tensor(from_l0), torch.tensor(to_l0)), num_nodes=cluster_size + 2)
    gender_dgl1 = dgl.graph((torch.tensor(from_l1), torch.tensor(to_l1)), num_nodes=cluster_size + 2)
    dgl.save_graphs(name_prev + ".dgl", [gender_dgl0, gender_dgl1])

