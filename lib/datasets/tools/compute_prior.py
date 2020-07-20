import numpy as np
import pickle
import os
import sys

NUM_ATTR_REL = 200
def cout_w(prob, num=NUM_ATTR_REL,dim=1):
    prob_weight = prob[:, :num]
    sum_value = np.sum(prob_weight, keepdims=True, axis=dim) + 0.1
    prob_weight = prob_weight / np.repeat(sum_value, prob_weight.shape[dim], axis=dim)
    return prob_weight

def cp_kl(a, b):
    # compute kl diverse
    if np.sum(a) == 0 or np.sum(b) == 0:
        return 1
    sum_ = a * np.log(a / b)
    all_value = [x for x in sum_ if str(x) != 'nan' and str(x) != 'inf']
    kl = np.sum(all_value)
    return kl

def compute_js(attr_prob):
    cls_num = attr_prob.shape[0]
    similarity = np.zeros((cls_num, cls_num))
    similarity[0, 1:] = 1
    similarity[1:, 0] = 1
    for i in range(1, cls_num):
        if i % 50 == 0:
            print('had proccessed {} cls...\n'.format(i))
        for j in range(1, cls_num):
            if i == j:
                similarity[i,j] = 0
            else:
                similarity[i,j] = 0.5 * (cp_kl(attr_prob[i, :], 0.5*(attr_prob[i, :] + attr_prob[j,:]))
                                         + cp_kl(attr_prob[j, :], 0.5*(attr_prob[i, :] + attr_prob[j, :])))
    return similarity

if __name__=='__main__':
    data_path = '/data/VisualGenome/graph/'
    dim_ = 1000
    ## Compute attribute knowledge by JS-diversion
    graph_a = pickle.load(open(data_path + 'vg_attr_frequency_1000.pkl', 'rb'))

    ## You can get part of graph_a and match name with your datasets
    #  We give an example of compute graph of VisualGenome with 1000 classes
    #  first line of graph_a is background
    graph_a = cout_w(graph_a, num=len(graph_a))
    graph_a = compute_js(graph_a)
    graph_a = 1 - graph_a
    pickle.dump(graph_a, open(data_path + 'vg_graph_a.pkl', 'wb'))

    ## Compute relation knowledge
    graph_r = pickle.load(open(data_path + 'vg_pair_frequency_1000.pkl', 'rb'))
    ## You can get part of graph_a and match name with your datasets
    #  We give an example of compute graph of VisualGenome with 1000 classes
    relation_matrix = np.zeros((dim_, dim_))
    relation_matrix = graph_r + graph_r.transpose()
    relation_matrix_row_sum = relation_matrix.sum(1)
    for i in range(dim_):
        relation_matrix[i, i] = relation_matrix_row_sum[i] + 1.
    prob_relation_matrix = np.zeros((dim_, dim_))
    for i in range(dim_):
        for j in range(dim_):
            prob_relation_matrix[i, j] = relation_matrix[i, j] / (
                        np.sqrt(relation_matrix[i, i]) * np.sqrt(relation_matrix[j, j]))
    prob_relation_matrix_ba = np.zeros((dim_ + 1, dim_ + 1))
    prob_relation_matrix_ba[1:, 1:] = prob_relation_matrix
    print(prob_relation_matrix_ba.shape)
    pickle.dump(prob_relation_matrix_ba, open(data_path + 'vg_graph_r.pkl', 'wb'))
