'''
this file tests the cluster metrics of topics as labels directly for different number of topics
'''
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import rand_score

# Calculates purity between two sets of labels
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    print(contingency_matrix)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

'''
Given a model type trained on a dataset, use topics as labels first to 
calculate the metrics 
'''
def calculate_metrics(model_type, dataset_name, num_topics):
    with open('./Model/{}_{}.pkl'.format(model_type, num_topics), 'rb') as inp:
        loaded_model = pickle.load(inp)

    doc_topic_probability = loaded_model['doc_topic_probas']


    '''
    Get the topic labels for each document
    '''
    topic_labels = [np.argmax(ele) for ele in doc_topic_probability]

    '''
    Get the groundtruth labels
    '''
    df = pd.read_json(dataset_name)
    sub_labels = df.sub_labels.values.tolist()

    purity = purity_score(sub_labels, topic_labels)
    rand_index = rand_score(sub_labels, topic_labels)
    nmi = normalized_mutual_info_score(sub_labels, topic_labels)

    print('*' * 20)
    print('Model {}; Topic {}'.format(model_type, num_topics))
    print('Purity', purity)
    print('Rand Index', rand_index)
    print('Normalized Mutual Information', nmi)
    print('*' * 20)


calculate_metrics('CTM', './Data/newsgroups/newsgroup_test.json', 100)
