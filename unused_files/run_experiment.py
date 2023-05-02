import time
from scipy.sparse import vstack
from spacy_topic_model import TopicModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from alto_session import NAITM
from sklearn.metrics import accuracy_score
import sys, os
from tomotopy import TermWeight
import pickle

USE_PREVIOUS_RESULT = False

file_name = './Data/newsgroup_sub_500.json'
save_json_file_name = "./Data/{}_newsgroup_sub_500_data.json"
dataset_name = 'newsgroup_sub_500'
test_dataset_name = './Data/newsgroup_sub_1000.json'

df = pd.read_json(file_name)
print(len(np.unique(df['label'])))
actual_labels = df.label.values.tolist()

from plotnine import *
def plot(data, xlabel, ylabel, title):
    data_long = pd.melt(data, id_vars=['k'], var_name='model', value_name='perplexity')

    # Create line plot
    return ggplot(data_long, aes(x='k', y='perplexity', color='model', group='model')) \
        + geom_line() \
        + xlab(xlabel) \
        + ylab(ylabel) \
        + ggtitle(title) \
        + theme_minimal()

def train_models(path_name, num_topics):
    SLDA_model = TopicModel(corpus_path=path_name, model_type='SLDA', min_num_topics= 5, num_iters= 600, load_model=True, save_model=True, load_path='./Model/model_data.pkl', hypers = None)
    LLDA_model = TopicModel(corpus_path=path_name, model_type='LLDA', min_num_topics= 5, num_iters= 600, load_model=True, save_model=True, load_path='./Model/model_data.pkl', hypers = None)
    LDA_model = TopicModel(corpus_path=path_name, model_type='LDA', min_num_topics= 5, num_iters= 600, load_model=True, save_model=True, load_path='./Model/model_data.pkl', hypers = None)
    models = dict()
    models['SLDA'] = SLDA_model
    models['LLDA'] = LLDA_model
    models['LDA'] = LDA_model
#     model_topics = dict()
    
    for key, model in models.items():
        print('start {} model'.format(key))
        model.preprocess(5, 100);
        start = time.time()
        model.train(num_topics);
        end = time.time()
        print('Took {} seconds to prepare the model'.format(end-start))
#         model_topics[key] = model.print_topics();
        
    return models


def get_topic_probability(model_collection):
    doc_distribution = dict()
    for key, model in model_collection.items():
        doc_distribution[key] = dict()
        start = time.time()
        document_probas, doc_topic_probas = model.group_docs_to_topics()
        end = time.time()
        print('Took {} seconds to group the model'.format(end-start))
        
        doc_distribution[key]['document_probas'] = document_probas
        doc_distribution[key]['doc_topic_probas'] = doc_topic_probas
        
    return doc_distribution

def get_accuracy(model_type, model_collection, document_data):
    result_df = model_collection[model_type].predict_labels(document_data)
    for k, v in result_df['topic_model_prediction_score'].items():
        result_df['topic_model_prediction_score'][k] = str(v)
        
    accuracy = sklearn.metrics.accuracy_score(result_df['label'], result_df['topic_model_prediction'])
    predicted_labels = result_df['topic_model_prediction']
    
    return accuracy, predicted_labels

def map_labels(model_type, model_collection, labels):
    simple_label_map = dict()

    unique_set = model_collection[model_type].label_set
    for i in range(len(unique_set)):
        simple_label_map[unique_set[i]] = i
        
    map_labels = []

    for ele in labels:
        map_labels.append(simple_label_map[ele])
    
    return simple_label_map, map_labels

def vectorize_texts(input_texts):
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
    vectorizer_idf = vectorizer.fit_transform(input_texts)
    
    return vectorizer_idf

def get_reverse_topics(model_type, model_collection):
    if model_type == 'LDA':
        simple_label_map, LDA_map_labels = map_labels(model_type, model_collection, model_collection[model_type].labels)
        return simple_label_map
    
    topics = model_collection[model_type].print_topics(verbose=False)
    
    topic_map = dict()
    for k, v in topics.items():
        topic_map[k] = v[1]

    reverse_topic_map = dict()
    for k, v in topics.items():
        reverse_topic_map[v[1]] = k
        
    return reverse_topic_map

def concatenate_input_vectorizer(processed_texts, model_collection, model_type, topics):
    result = []
    for i, ele in enumerate(processed_texts):
        top_words = model_collection[model_type].predict_doc(i, 1)
        topic_num = top_words[0][0]
        
        if model_type == 'LDA':
            topic_words = topics[topic_num]
        else:
            topic_words = topics[topic_num][0]
#         print('top words are ')
#         print(top_words)
        concatanated_texts = ele + topic_words
        if len(concatanated_texts) == 0:
            print(i)
        
        result.append(' '.join(concatanated_texts))
        
    return result

def predict_labels(model_collection, model_type):
    inferred, _ = model_collection[model_type].lda_model.infer(model_collection[model_type].corpus)
    # print(inferred)
    preds = model_collection[model_type].lda_model.estimate(inferred)
    
    label_predictions = []
    for i, scores in enumerate(preds):
        topic_num = int(np.argmax(scores))
        label_predictions.append(model_collection[model_type].label_set[topic_num])
        
    return label_predictions

def simulate_experiment(model_collection, model_type, model_topic_distribution, dataframe, test_path, break_down_test_size, input_vectorizer):
    topics = model_collection[model_type].print_topics(verbose=False)
    test_df = pd.read_json(test_path)
    
    '''
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
    token_lst = concatenate_input_vectorizer(model_collection[model_type].data_words_nonstop, model_collection, model_type, topics)
    input_vectorizer = vectorizer.fit_transform(token_lst)
    '''
    
#     test_texts = test_df['text']
    test_labels = test_df['label']
#     test_vectorizer_idf = vectorizer.fit_transform(test_texts)
    
    simple_label_map, model_label_map  = map_labels(model_type, model_collection, actual_labels)
    
    test_map_labels = []
    for ele in test_labels:
        test_map_labels.append(simple_label_map[ele])
    
    
    reverse_label_mapping = get_reverse_topics(model_type, model_collection)
    
#     print('simple label map for {} is {}'.format(model_type, simple_label_map))
#     print('reverse label map for {} is {}'.format(model_type, reverse_label_mapping))
#     return
    
    gold_labels = model_collection[model_type].labels
    
    session = NAITM(model_collection[model_type].get_texts(), model_topic_distribution[model_type]['document_probas'],  model_topic_distribution[model_type]['doc_topic_probas'], dataframe, 'logreg', input_vectorizer, len(topics), 500)
    
    
    doc_count = 0
    doc_recommend_score = []
    lst = []
    acc= []
    document_stack, labels_track = None, []
    classifier_acc = []
    recommend_ids1, model_inferred_topics1, actual_topic1 = [], [], []

    test_acc1 = dict()
    for ele in break_down_test_size:
        test_acc1[ele] = []

    logreg_model3 = SGDClassifier(loss="log", penalty="l2", max_iter=1000, tol=1e-3, random_state=42, learning_rate="adaptive", eta0=0.1, validation_fraction=0.2)
    start_time = time.time()
    while doc_count < 500:
        try:
            random_document, score = session.recommend_document();
        except:
            print('reached the end of the experiment. {} docs labeled'.format(doc_count))
            end_time = time.time()
            break
        
    #     inferred_topics = models['SLDA'].predict_doc(random_document, 3);

    #     inferred_topics = slda_predicted[random_document]
        inferred_topics = gold_labels[random_document]

        if document_stack == None:
            document_stack = input_vectorizer[random_document]
        else:
            document_stack = vstack((document_stack, input_vectorizer[random_document]))
        
#         print('reverse mapping is {}'.format(reverse_label_mapping))
#         print('gola label is {}'.format(gold_labels[random_document]))
        
    #     labels_track.append(reverse_SLDA_topic_map[slda_predicted[random_document]])
        labels_track.append(reverse_label_mapping[gold_labels[random_document]])

        logreg_model3.partial_fit(document_stack, labels_track, list(range(len(model_collection[model_type].label_set))))
        logreg_y_pred= logreg_model3.predict(input_vectorizer[0:500])



        accuracy = accuracy_score(test_map_labels[0:500], logreg_y_pred)

        for j in break_down_test_size:
    #         print(j)
            test_logreg_y_pred= logreg_model3.predict(input_vectorizer[500:j])
            test_accuracy = accuracy_score(test_map_labels[500:j], test_logreg_y_pred)
            test_acc1[j].append(test_accuracy)

        print('test set acc {}'.format(test_acc1[800][-1]))
        '''
        Save this part
        '''
        classifier_acc.append(accuracy)
        doc_recommend_score.append(score)


        '''
        remember to also save recommended document id and inferred topics
        '''
        recommend_ids1.append(random_document)
        model_inferred_topics1.append(inferred_topics[0][0])
        actual_topic1.append(reverse_label_mapping[gold_labels[random_document]])

        print('document id {}'.format(random_document))
        print('inferred topics are {}'.format(inferred_topics))
        print('classifier accuracy is {}'.format(accuracy))
        print('\033[1mactual label is {}\033[0m'.format(model_collection[model_type].labels[random_document]))
    #     session.label(random_document, user_input)
    #     print('\033[1mpredicted topic is {}\033[0m'.format(reverse_SLDA_topic_map[slda_predicted[random_document]]))
        print('\033[1mgold topic is {}\033[0m'.format(reverse_label_mapping[gold_labels[random_document]]))

    #     session.label(random_document, reverse_SLDA_topic_map[slda_predicted[random_document]])
        session.label(random_document, reverse_label_mapping[gold_labels[random_document]])

    #     logreg_y_pred= .predict(vectorizer_idf)

        lst.append(doc_count)
        
        if doc_count % 7 == 0 and model_type == 'SLDA':
            model_collection[model_type].retrain()
        
        doc_count += 1
        end_time = time.time()
    
    result = pd.DataFrame()
    result['classifier acc'] = classifier_acc
    result['inferred topic'] = model_inferred_topics1
    result['actual topic'] = actual_topic1
    result['recommend ids'] = recommend_ids1
    result['recommend score'] = doc_recommend_score

    for k, v in test_acc1.items():
        result['{} test set'.format(k-500)] = v
    
    result.to_csv('./Data/{}_result.csv'.format(model_type), index=False)
    
    return result, end_time-start_time

experiment_models = train_models(file_name, 20)

experiment_topic_distribution = get_topic_probability(experiment_models)

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1,2))
experiment_vectorizer_idf = vectorizer.fit_transform(pd.read_json(test_dataset_name)['text'])

experiment_test_sizes = [i for i in range(len(pd.read_json(test_dataset_name)['text']) + 1) if i > 500 and i % 100 == 0]

# Check the accuracy rate of sLDA
sLDA_pred = predict_labels(experiment_models, 'SLDA')
print(sklearn.metrics.accuracy_score(experiment_models['SLDA'].labels, sLDA_pred))

SLDA_experiment_result, SLDA_experiment_time = simulate_experiment(experiment_models, 'SLDA', experiment_topic_distribution, df, test_dataset_name, experiment_test_sizes, experiment_vectorizer_idf)

if os.path.exists('./Model/LLDA_model_{}_result.pkl'.format(dataset_name)) and USE_PREVIOUS_RESULT:
    print('exists')
    with open('./Model/LLDA_model_{}_result.pkl'.format(dataset_name), 'rb') as inp:
        LLDA_experiment_result = pickle.load(inp)
else:
    print('not exists')
    LLDA_experiment_result, LLDA_experiment_time = simulate_experiment(experiment_models, 'LLDA', experiment_topic_distribution, df, test_dataset_name, experiment_test_sizes, experiment_vectorizer_idf)
    with open('./Model/LLDA_model_{}_result.pkl'.format(dataset_name), 'wb+') as outp:
        pickle.dump(LLDA_experiment_result, outp)

if os.path.exists('./Model/LDA_model_{}_result.pkl'.format(dataset_name)) and USE_PREVIOUS_RESULT:
    with open('./Model/LDA_model_{}_result.pkl'.format(dataset_name), 'rb') as inp:
        LDA_experiment_result = pickle.load(inp)
else:
    LDA_experiment_result, LDA_experiment_time = simulate_experiment(experiment_models, 'LDA', experiment_topic_distribution, df, test_dataset_name, experiment_test_sizes, experiment_vectorizer_idf)
    with open('./Model/LDA_model_{}_result.pkl'.format(dataset_name), 'wb+') as outp:
        pickle.dump(LDA_experiment_result, outp)


    

SLDA_experiment_acc = SLDA_experiment_result['classifier acc']
LLDA_experiment_acc = LLDA_experiment_result['classifier acc']
LDA_experiment_acc = LDA_experiment_result['classifier acc']

lst_len = min(len(SLDA_experiment_acc), len(LLDA_experiment_acc), len(LDA_experiment_acc))

# Create data frame from the given data
data = pd.DataFrame({
    'k': [i for i in range(lst_len)],
    'LLDA': LLDA_experiment_acc[0:lst_len],
    'SLDA': SLDA_experiment_acc[0:lst_len],
    'LDA': LDA_experiment_acc[0:lst_len]
})

plot_res = plot(data, 'Number of documents labeled (k)', 'Training Set Accuracy', 'Training Accuracy of LDA, LLDA and SLDA')
ggsave(plot_res,file='Training_accuracy.png')

SLDA_experiment_test_acc = SLDA_experiment_result['400 test set']
LLDA_experiment_test_acc = LLDA_experiment_result['400 test set']
LDA_experiment_test_acc = LDA_experiment_result['400 test set']

# Create data frame from the given data
data = pd.DataFrame({
    'k': [i for i in range(lst_len)],
    'LLDA': LLDA_experiment_test_acc[0:lst_len],
    'SLDA': SLDA_experiment_test_acc[0:lst_len],
    'LDA': LDA_experiment_test_acc[0:lst_len]
})

plot_res1 = plot(data, 'Number of documents labeled (k)', 'Test Set Accuracy', 'Testing Accuracy of LDA, LLDA and SLDA (Test Size 400)')
ggsave(plot_res1,file='Testing_accuracy_400.png')

# Create data frame from the given data
data = pd.DataFrame({
    'k': [i for i in range(lst_len)],
    'LLDA': LLDA_experiment_result['100 test set'][0:lst_len],
    'SLDA': SLDA_experiment_result['100 test set'][0:lst_len],
    'LDA': LDA_experiment_result['100 test set'][0:lst_len]
})

plot_res2 = plot(data, 'Number of documents labeled (k)', 'Test Set Accuracy', 'Testing Accuracy of LDA, LLDA and SLDA (Test Size 100)')
ggsave(plot_res2,file='Testing_accuracy_150.png')