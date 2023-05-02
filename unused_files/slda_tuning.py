import pickle
import sklearn.metrics as metric
import tomotopy as tp
import numpy as np
import pandas as pd
import statistics

def predict_labels(model):
    inferred, _ = model.infer(corpus)
    # print(inferred)
    preds = model.estimate(inferred)
    
    label_predictions = []
    for i, scores in enumerate(preds):
        topic_num = int(np.argmax(scores))
        label_predictions.append(label_set[topic_num])
        
    return label_predictions


df = pd.read_json('./Data/newsgroup_sub_500.json')
load_path = './Model/SLDA_model_newsgroup_sub_500.pkl'
with open(load_path, 'rb') as inp:
    read_data = pickle.load(inp)
            
    data_words_nonstop = read_data['data_words_nonstop']
    label_set = read_data['label_set']
    corpus = read_data['corpus']
    word_spans = read_data['word_spans']

accuracies4 = []
for i in range(10):
    print('training model {}'.format(i+1))
    mdl3 = tp.SLDAModel(k=20, vars=['b' for _ in range(20)],  mu = [1.1 for _ in range(len(label_set))], glm_param= [1.1 for i in range(len(label_set))], nu_sq = [5 for i in range(len(label_set))])
    mdl3.add_corpus(corpus)
#     for i in range(0, 500, 10):
        # print('training iter {}'.format(i))
    mdl3.train(800)

    SLDA_df3 = predict_labels(mdl3)
#     print(sklearn.metrics.accuracy_score(SLDA_df2, df['label'].tolist()))
    accuracy = metric.accuracy_score(SLDA_df3, df['label'].tolist())
    print('Accuracy score: {}'.format(accuracy))
    accuracies4.append(accuracy)

print(statistics.mean(accuracies4))
print(statistics.median(accuracies4))
print(max(accuracies4))